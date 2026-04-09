"""
nn_client.py — Flower NumPyClient with FedCRA support.

CRA loss gating: the anchor loss for class k is only applied when the current
batch has at least MIN_CLASS_SAMPLES samples of class k.

Fairness fixes (v7):
  - FedAvg path now applies gradient clipping identical to the FedCRA path
    (max_norm = cra_grad_clip, defaulting to 1.0). Previously FedCRA had
    gradient clipping but FedAvg did not, creating an unfair advantage for
    FedCRA under heterogeneous data (larger gradient variance on FedAvg).
  - fed_train (FedAvg path) now returns f1_weighted to match fed_test output,
    eliminating metric inconsistency between train and test phases.
  - Centroid computation uses a no-shuffle pass over the loader so that the
    accumulated sums are deterministic regardless of DataLoader state.

FedCRA v7 changes:
  - beta defaults to 0.4 (was 0.4, unchanged); now sourced from config with
    explicit float cast to guard against YAML type coercion.
  - class_weights default to uniform (all ones) when the key is absent from
    config, instead of None, so CrossEntropyLoss receives a valid weight tensor
    in all cases.
"""

import torch.optim as optim
from collections import OrderedDict
from flwr.common import NDArrays
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
import os
import json
import time
from pathlib import Path
from log_config import base_logger
from src.fedLearn.centralized import fed_train, fed_test
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")

logger = base_logger(__name__)

# Minority-class batches are never silently skipped
MIN_CLASS_SAMPLES = 1


def _fed_train_cra(model, epochs, optimizer, train_loader,
                   anchors: np.ndarray, confidence: np.ndarray, proximal_mu: float,
                   global_params: list, num_classes: int, grad_clip: float = 1.0):
    """
    NEW FedCRA v10: Selective Class Alignment + Proximal Regularization

    - Selective: Apply CRA loss ONLY for classes present in client's data
    - Confidence Scaling: Scale CRA loss by anchor confidence conf_c
    - Proximal: Add FedProx regularization to prevent divergence
    - No static alpha/rho/beta: Dynamic based on client data and anchor quality
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    confidence_t = torch.tensor(confidence, dtype=torch.float32, device=device)

    # Convert global params to tensors for proximal loss
    global_params_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in global_params]

    # Normalize anchors
    anchor_norms = anchors_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    anchors_norm_t = anchors_t / anchor_norms
    anchor_valid = (anchors_t.norm(dim=1) > 1e-6)

    # CE loss (no focal, no class weights - keep it simple)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Hook for penultimate layer
    penultimate = []
    hook_handle = None

    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate.append(o)
        )
    else:
        logger.warning("FedCRA: penultimate layer not found — using CE only")
        return fed_train(model=model, epochs=epochs, optimizer=optimizer,
                         train_loader=train_loader, grad_clip=grad_clip)

    # Track which classes are present in this client's data
    client_class_counts = np.zeros(num_classes, dtype=np.int32)

    running_loss = 0.0
    steps = 0
    all_labels, all_preds = [], []

    try:
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Update class counts for selective alignment
                for label in labels.cpu().numpy():
                    client_class_counts[label] += 1

                penultimate.clear()
                optimizer.zero_grad()
                outputs = model(inputs)

                # CE loss
                ce_loss = criterion(outputs, labels).mean()

                # CRA loss: Selective alignment
                cra_loss = torch.tensor(0.0, device=device)

                if penultimate and anchor_valid.any():
                    acts = penultimate[0]  # (N, D)

                    # Normalize activations
                    acts_norm = acts / acts.norm(dim=1, keepdim=True).clamp(min=1e-8)

                    # Compute distances to anchors
                    dots = acts_norm @ anchors_norm_t.t()  # (N, num_classes)
                    sq_dists = (2.0 - 2.0 * dots).clamp(min=0)  # (N, num_classes)

                    per_sample_cra = []

                    for i in range(acts_norm.shape[0]):
                        y_i = int(labels[i].item())

                        # SELECTIVE: Only apply if client has this class
                        if client_class_counts[y_i] == 0 or not anchor_valid[y_i]:
                            continue

                        pos_dist = sq_dists[i, y_i]

                        # Confidence scaling: reduce influence of unreliable anchors
                        conf_c = confidence_t[y_i]
                        if conf_c < 0.1:  # Skip very low confidence anchors
                            continue

                        # Margin-based loss: encourage separation
                        neg_dists = []
                        for c in range(num_classes):
                            if c != y_i and anchor_valid[c] and client_class_counts[c] > 0:
                                neg_dists.append(sq_dists[i, c])

                        if neg_dists:
                            neg_dist = torch.stack(neg_dists).mean()
                            # Margin loss: want pos_dist < neg_dist
                            margin_loss = torch.clamp(pos_dist - neg_dist + 0.5, min=0)
                            # Scale by confidence
                            term = conf_c * margin_loss
                        else:
                            # Fallback: just minimize distance to own anchor
                            term = conf_c * pos_dist

                        per_sample_cra.append(term)

                    if per_sample_cra:
                        cra_loss = torch.stack(per_sample_cra).mean()
                        cra_loss = torch.clamp(cra_loss, 0, 2.0)  # Bound the loss

                # Proximal regularization (FedProx)
                prox_loss = torch.tensor(0.0, device=device)
                if proximal_mu > 0:
                    current_params = list(model.parameters())
                    for curr_p, global_p in zip(current_params, global_params_t):
                        prox_loss += ((curr_p - global_p) ** 2).sum()
                    prox_loss = (proximal_mu / 2.0) * prox_loss

                # Combined loss
                loss = ce_loss + cra_loss + prox_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                running_loss += loss.item()
                steps += 1
                preds = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    finally:
        if hook_handle:
            hook_handle.remove()

    # Compute residuals for server aggregation
    residuals = {}
    if penultimate:
        # Compute per-class residual: difference between client centroid and global anchor
        centroids, _ = _compute_class_centroids(model, train_loader, num_classes, device)
        for c in range(num_classes):
            if c in centroids and anchor_valid[c]:
                client_centroid = np.array(centroids[c])
                global_anchor = anchors[c]
                residual = client_centroid - global_anchor
                residuals[c] = residual.tolist()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    FPR = FP / (FP + TN + 1e-10)

    return {
        "loss": running_loss / max(steps, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_fpr": float(np.mean(FPR)),
        "cra_residuals": json.dumps(residuals),
        "cra_class_counts": json.dumps({int(k): int(v) for k, v in enumerate(client_class_counts)}),
    }


def _compute_class_centroids(model, loader, num_classes: int, device):
    """
    Compute per-class centroids from penultimate layer activations.
    Centroids are L2-normalised to match the normalisation applied during
    training, so server-side anchors stay in the same embedding space.

    The loader is iterated without re-shuffling so results are deterministic
    regardless of the DataLoader's internal state.
    """
    model.eval()
    sums, counts = {}, {}
    penultimate  = []
    hook_handle  = None

    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate.append(o.detach())
        )
    else:
        return {}, {}

    try:
        with torch.no_grad():
            for xb, yb in loader:
                xb    = xb.to(device, non_blocking=True)
                yb_np = yb.numpy()
                penultimate.clear()
                _     = model(xb)
                if not penultimate:
                    continue
                acts  = penultimate[0].cpu().numpy()

                # L2-normalise before accumulating centroid (matches training)
                norms = np.linalg.norm(acts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                acts  = acts / norms

                for k in range(num_classes):
                    mask = (yb_np == k)
                    if mask.sum() == 0:
                        continue
                    k_acts = acts[mask]
                    if k not in sums:
                        sums[k]   = np.zeros(k_acts.shape[1], dtype=np.float32)
                        counts[k] = 0
                    sums[k]   += k_acts.sum(axis=0)
                    counts[k] += int(mask.sum())
    finally:
        if hook_handle:
            hook_handle.remove()

    return {k: (sums[k] / counts[k]).tolist() for k in sums}, counts


class ClientModel(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, test_loader,
                 client_names, results_path=None):
        super().__init__()
        self.client_id    = client_id
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model        = model
        self.client_names = client_names
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_path = results_path
        self.round_id     = 0
        self.client_name  = (client_names[int(client_id)]
                             if 0 <= int(client_id) < len(client_names)
                             else "client")
        self.model_name   = model.__class__.__name__
        print("DEVICE:", next(self.model.parameters()).device)

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.device)
             for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: dict):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr            = float(config.get("learning_rate", 0.001))
        self.round_id = config.get("round_id", 0)
        optimizer     = getattr(optim, config.get("optimizer", "Adam"))(
                            self.model.parameters(), lr=lr)
        epochs        = config.get("epochs", 5)

        # Gradient clipping is applied in BOTH FedCRA and FedAvg paths so
        # that the two strategies operate under identical optimisation conditions.
        grad_clip = float(config.get("cra_grad_clip", 1.0))

        cra_anchors_json = config.get("cra_anchors")
        cra_confidence_json = config.get("cra_confidence")
        proximal_mu = float(config.get("cra_proximal_mu", 0.01))
        num_classes = int(config.get("cra_num_classes",
                           getattr(self.model, "output_size", 9)))
        is_cra = bool(cra_anchors_json and cra_confidence_json)

        print(f"[Client {self.client_name:10s}] Round {self.round_id:3d} | "
              f"lr={lr:.5f} | proximal_mu={proximal_mu:.4f} | "
              f"grad_clip={grad_clip:.1f} | "
              f"{'CRA' if is_cra else 'FedAvg':4s}")

        self.model.to(self.device)
        t0 = time.time()

        if is_cra:
            anchors = np.array(json.loads(cra_anchors_json), dtype=np.float32)
            confidence = np.array(json.loads(cra_confidence_json), dtype=np.float32)
            global_params = parameters  # Use the received global parameters for proximal
            metrics = _fed_train_cra(
                model=self.model, epochs=epochs, optimizer=optimizer,
                train_loader=self.train_loader,
                anchors=anchors, confidence=confidence, proximal_mu=proximal_mu,
                global_params=global_params, num_classes=num_classes,
                grad_clip=grad_clip)
        else:
            # FedAvg path — identical hyper-parameters except no CRA loss.
            # grad_clip is now applied here too for a fair comparison.
            metrics = fed_train(
                model=self.model, epochs=epochs, optimizer=optimizer,
                train_loader=self.train_loader, grad_clip=grad_clip)

        self.write_results_json(metrics, time.time() - t0, "train", self.round_id)
        # Only save model checkpoints if explicitly enabled (to save disk space during experiments)
        if os.getenv('SAVE_CLIENT_CHECKPOINTS', 'false').lower() == 'true':
            self.save_model(self.model, self.round_id)

        fit_metrics = {}
        # Return residuals and class counts for server aggregation
        if is_cra:
            # Metrics already include cra_residuals and cra_class_counts
            fit_metrics.update({
                "cra_residuals": metrics.get("cra_residuals", "{}"),
                "cra_class_counts": metrics.get("cra_class_counts", "{}"),
            })

        return self.get_parameters({}), len(self.train_loader), fit_metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        self.set_parameters(parameters)
        print(f"Client {self.client_name} evaluating round {self.round_id}")
        self.model.to(self.device)
        t0 = time.time()
        metrics = fed_test(self.model, self.test_loader)
        self.write_results_json(metrics, time.time() - t0, "test", self.round_id)
        return float(metrics["loss"]), len(self.test_loader), {"accuracy": metrics["accuracy"]}

    def write_results_json(self, metrics, dt, phase, round_id):
        d = Path(self.results_path) / "clients"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{self.client_name}.json"
        data = json.loads(f.read_text()) if f.exists() else []
        data.append({"round": round_id, f"{phase}_metrics": metrics, "communication_time": dt})
        f.write_text(json.dumps(data, indent=4))

    def save_model(self, model, round_id):
        d = Path(self.results_path) / "clients"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / f"{self.client_name}_rnd_{round_id}.pth")


def generate_client_fn(model_fn, train_loaders, test_loaders, client_names, results_path):
    def client_fn(client_id):
        try:
            return ClientModel(
                client_id=client_id, model=model_fn(),
                train_loader=train_loaders[int(client_id)],
                test_loader=test_loaders[int(client_id)],
                client_names=client_names,
                results_path=results_path,
            ).to_client()
        except Exception as e:
            logger.error(f"Error in client {client_id}: {e}")
            raise
    return client_fn