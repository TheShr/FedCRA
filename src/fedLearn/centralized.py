"""
centralized.py  —  FedCRA v11
==============================

fed_train() is the central function that orchestrates:
  1. Single forward pass for both CE loss and embedding extraction
  2. CRA penalty (passes embeddings, no second forward pass)
  3. Class-conditional proximal regularisation (separate from CRA)
  4. Gradient clipping (fixes gradient explosion on imbalanced CIC-IoMT)
  5. Returns cra_data with properly JSON-serialisable values

FIXES vs v10
------------
[CRITICAL] Single forward pass.
    Old: model(inputs) for CE, then CRALoss._extract_embeddings() again.
    New: model.forward_with_embeddings(inputs) once → (logits, embeddings).

[HIGH] Gradient clipping applied.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    Uses cra_grad_clip from config (default 1.0).

[HIGH] cra_data is JSON-serialisable.
    cra_residuals: dict[str, list]  (numpy arrays → lists)
    cra_class_counts: dict[str, int]
    client_id: str
    All scalar/string/list — Flower can serialize without crashing.

[HIGH] client_id included in returned cra_data.
    Server uses this to key client_weights reliably.

[MEDIUM] Proximal term uses class_mu from server (separate from CRA loss).
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .fedcra_loss import CRALoss, compute_proximal_loss


def forward_with_embeddings(model, inputs):
    """
    Single forward pass returning both logits and L2-normalised embeddings.

    Hooks fc_layers[-2] (the _EmbeddingHead) output, which is what
    FedCRA's anchors are defined against.
    """
    x = inputs.view(-1, model.input_size)
    for layer in model.fc_layers[:-1]:     # up to but not including classifier
        x = layer(x)
    embeddings = F.normalize(x, p=2, dim=1)
    logits     = model.fc_layers[-1](x)    # classifier head
    return logits, embeddings


def fed_train(
    model,
    epochs: int,
    optimizer,
    train_loader,
    cra_config: dict | None = None,
    client_id: str | int   = "unknown",
    grad_clip: float        = 1.0,
):
    """
    Federated local training for one round.

    Parameters
    ----------
    model        : DNN (or any model with fc_layers[-2] embedding head)
    epochs       : number of local epochs
    optimizer    : torch.optim instance
    train_loader : DataLoader
    cra_config   : dict from server (contains cra_anchors, cra_mu_c, etc.)
                   None → plain CE training (FedAvg behaviour)
    client_id    : str/int identifier — returned in cra_data for server routing
    grad_clip    : max gradient norm (config.cra_grad_clip, default 1.0)

    Returns
    -------
    metrics  : dict with loss, accuracy, macro_f1, weighted_f1, precision, recall
    cra_data : dict with JSON-serialisable cra_residuals, cra_class_counts,
               client_id  —  or None if cra_config is None
    """
    device    = next(model.parameters()).device
    
    # Setup class-weighted CE loss only when CRA / explicit class-weight config is active.
    criterion = nn.CrossEntropyLoss()
    if cra_config and (
        "cra_anchors" in cra_config
        or cra_config.get("cra_use_class_penalty", False)
    ):
        # Prefer explicit CRA class count, otherwise infer from model output.
        model_num_classes = None
        if hasattr(model, "output_size"):
            model_num_classes = int(model.output_size)
        elif hasattr(model, "fc_layers") and len(model.fc_layers) > 0:
            model_num_classes = int(getattr(model.fc_layers[-1], "out_features", 0))

        num_classes = int(cra_config.get("cra_num_classes", model_num_classes or 5))
        if model_num_classes and num_classes != model_num_classes:
            num_classes = model_num_classes

        class_counts = np.zeros(num_classes, dtype=np.int32)
        for inputs, labels in train_loader:
            for label in labels.numpy():
                if 0 <= label < num_classes:
                    class_counts[label] += 1
        
        total_samples = sum(class_counts)
        if total_samples > 0:
            class_weights = np.ones(num_classes, dtype=np.float32)
            for c in range(num_classes):
                if class_counts[c] > 0:
                    # Inverse frequency weighting (sqrt for moderate scaling)
                    freq = class_counts[c] / total_samples
                    class_weights[c] = np.sqrt(1.0 / (freq + 1e-8))
            
            # Normalize to mean = 1.0
            valid_weights = class_weights[class_counts > 0]
            if len(valid_weights) > 0:
                class_weights = class_weights / (np.mean(valid_weights) + 1e-8)
            
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Build CRA loss object once per round (only if CRA-specific keys present)
    cra_loss_fn = None
    if cra_config and "cra_anchors" in cra_config:
        cra_loss_fn = CRALoss(cra_config)

    # Snapshot global parameters for proximal term
    global_params = None
    class_mu      = None
    if cra_config and ("cra_mu_c" in cra_config or "cra_proximal_mu" in cra_config):
        global_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
        }
        if "cra_mu_c" in cra_config:
            class_mu = torch.tensor(
                json.loads(cra_config["cra_mu_c"]),
                dtype=torch.float32, device=device
            )

    # Accumulators across all epochs
    all_preds, all_labels = [], []
    total_loss            = 0.0
    n_batches             = 0

    # Per-class residual accumulators (for server anchor update)
    residual_sums   = {}    # class_id -> sum of mean embedding vectors
    residual_counts = {}    # class_id -> number of batches that had this class
    class_counts    = {}    # class_id -> total sample count

    model.train()

    for _epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ── 1. SINGLE forward pass (fixes double-forward bug) ─────────
            logits, embeddings = forward_with_embeddings(model, inputs)

            # ── 2. Cross-entropy loss ──────────────────────────────────────
            ce_loss = criterion(logits, labels)

            # ── 3. CRA penalty (uses pre-computed embeddings) ─────────────
            lambda_cra = cra_config.get("lambda_cra", 0.1) if cra_config else 0.1
            cra_penalty = torch.tensor(0.0, device=device)
            if cra_loss_fn is not None:
                cra_penalty = lambda_cra * cra_loss_fn(embeddings=embeddings, labels=labels)

            # ── 4. Proximal regularisation (class-conditional or FedProx) ──
            prox_penalty = torch.tensor(0.0, device=device)
            if global_params is not None:
                if class_mu is not None:
                    prox_penalty = compute_proximal_loss(
                        model, global_params, class_mu, labels, device
                    )
                elif "cra_proximal_mu" in cra_config:
                    proximal_mu = float(cra_config["cra_proximal_mu"])
                    prox_penalty = sum(
                        (param - global_params[name]).pow(2).sum()
                        for name, param in model.named_parameters()
                        if name in global_params
                    ) * (proximal_mu / 2.0)

            loss = ce_loss + cra_penalty + prox_penalty
            loss.backward()

            # ── 5. Gradient clipping (fixes explosion on imbalanced data) ─
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            # ── Accumulate for metrics ─────────────────────────────────────
            total_loss += loss.item()
            n_batches  += 1
            preds       = logits.argmax(dim=1).cpu().numpy()
            lbls        = labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

            # ── Accumulate CRA residuals (for server anchor update) ────────
            if cra_loss_fn is not None:
                mean_embs, batch_counts = cra_loss_fn.compute_class_residuals(
                    embeddings, labels
                )
                for c, vec in mean_embs.items():
                    residual_sums[c]   = residual_sums.get(c, np.zeros_like(vec)) + vec
                    residual_counts[c] = residual_counts.get(c, 0) + 1
                for c, cnt in batch_counts.items():
                    class_counts[c] = class_counts.get(c, 0) + cnt

    # ── Metrics ────────────────────────────────────────────────────────────
    avg_loss   = total_loss / max(n_batches, 1)
    accuracy   = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro",     zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision  = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall     = recall_score(all_labels, all_preds, average="weighted",    zero_division=0)

    metrics = {
        "loss":         avg_loss,
        "accuracy":     accuracy,
        "macro_f1":     macro_f1,
        "weighted_f1":  weighted_f1,
        "precision":    precision,
        "recall":       recall,
    }

    # ── CRA data for server ────────────────────────────────────────────────
    cra_data = None
    if cra_loss_fn is not None:
        # Average residual vectors across batches
        mean_residuals = {
            c: (residual_sums[c] / residual_counts[c]).tolist()
            for c in residual_sums
            if residual_counts.get(c, 0) > 0
        }
        cra_data = {
            # JSON strings — Flower can serialise these as scalar string metrics
            "cra_residuals":   json.dumps({str(k): v for k, v in mean_residuals.items()}),
            "cra_class_counts": json.dumps({str(k): int(v) for k, v in class_counts.items()}),
            # client_id for server to key client_weights reliably (fixes proxy bug)
            "client_id":       str(client_id),
        }

    return metrics, cra_data


def fed_test(model, test_loader):
    """Evaluate model on local test set."""
    device    = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    n = max(len(test_loader), 1)
    return {
        "loss":         total_loss / n,
        "accuracy":     accuracy_score(all_labels, all_preds),
        "macro_f1":     f1_score(all_labels, all_preds, average="macro",     zero_division=0),
        "weighted_f1":  f1_score(all_labels, all_preds, average="weighted",  zero_division=0),
        "precision":    precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall":       recall_score(all_labels, all_preds, average="weighted",    zero_division=0),
    }


def compute_macro_fpr(y_true, y_pred):
    """
    Compute macro-averaged False Positive Rate (FPR) for multiclass classification.
    FPR = FP / (FP + TN) for each class, then average.
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fprs = []

    for i in range(n_classes):
        # True Positives, False Positives, False Negatives, True Negatives for class i
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # sum of column i except diagonal
        fn = cm[i, :].sum() - tp  # sum of row i except diagonal
        tn = cm.sum() - (tp + fp + fn)

        if fp + tn == 0:
            fpr = 0.0
        else:
            fpr = fp / (fp + tn)
        fprs.append(fpr)

    return np.mean(fprs)


def compute_per_class_f1(y_true, y_pred, num_classes=None):
    """
    Compute F1 score for each class.
    Returns a dict {class_id: f1_score}
    """
    from sklearn.metrics import f1_score
    import numpy as np

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.arange(int(max(y_true_arr.max(), y_pred_arr.max())) + 1)

    per_class_f1 = {}
    for cls in classes:
        y_true_bin = (y_true_arr == cls).astype(int)
        y_pred_bin = (y_pred_arr == cls).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        per_class_f1[int(cls)] = f1

    return per_class_f1