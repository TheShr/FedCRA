"""
FedCRA Client Training — Improved Implementation
=================================================

Implements the client-side FedCRA training with:
1. Selective class alignment (Component D)
2. Proper loss combination (CE + CRA + Proximal)
3. Class-balanced weighting
4. Efficient single-pass forward with embedding extraction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_fedcra(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    epochs: int,
    cra_config: dict,
    global_params: list,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    FedCRA client training with all four novel components.
    
    Loss Formulation:
    -----------------
    Total Loss = CE(y, ŷ) + λ_CRA × CRA(embeddings) + (μ_c/2) × ||θ - θ_g||²
    
    Where:
    - CE: Cross-entropy loss with class weighting
    - CRA: Anchor alignment loss (embedding space)
    - μ_c: Class-conditional proximal penalty (parameter space)
    
    Parameters:
    -----------
    model : nn.Module
        Local client model
        
    train_loader : DataLoader
        Training data for this client
        
    optimizer : optim.Optimizer
        Optimizer (SGD recommended)
        
    epochs : int
        Number of local training epochs
        
    cra_config : dict
        FedCRA configuration from server containing:
        - cra_anchors: Global class anchors [num_classes, embedding_dim]
        - cra_confidence: Anchor confidence scores [num_classes]
        - cra_mu_c: Class-conditional penalties [num_classes]
        - cra_num_classes: Number of classes
        - round_id: Current federated round
        
    global_params : list
        Global model parameters for proximal term
        
    grad_clip : float
        Gradient clipping norm
        
    device : torch.device
        Device for computation
    
    Returns:
    --------
    metrics : dict
        Training metrics including:
        - loss, accuracy, f1_score, etc.
        - cra_residuals: Per-class embeddings for server
        - cra_class_counts: Class distribution
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    
    # Parse configuration
    num_classes = int(cra_config["cra_num_classes"])
    round_id = int(cra_config.get("round_id", 0))
    
    # Parse global anchors and confidence
    anchors = torch.tensor(
        json.loads(cra_config["cra_anchors"]),
        dtype=torch.float32,
        device=device
    )  # [num_classes, embedding_dim]
    
    confidence = torch.tensor(
        json.loads(cra_config["cra_confidence"]),
        dtype=torch.float32,
        device=device
    )  # [num_classes]
    
    # Parse class-conditional penalties (μ_c)
    mu_c = torch.tensor(
        json.loads(cra_config["cra_mu_c"]),
        dtype=torch.float32,
        device=device
    )  # [num_classes]
    
    # Convert global params to tensors
    global_params_dict = {}
    for name, param in zip(model.state_dict().keys(), global_params):
        global_params_dict[name] = torch.tensor(param, dtype=torch.float32, device=device)
    
    # Normalize anchors for distance computation
    anchor_norms = anchors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    anchors_normalized = anchors / anchor_norms
    anchor_valid = (anchors.norm(dim=1) > 1e-6)
    
    # ══════════════════════════════════════════════════════════════════
    # COMPONENT D: Selective Class Alignment
    # ══════════════════════════════════════════════════════════════════
    # Track which classes are present in client's data
    client_class_counts = np.zeros(num_classes, dtype=np.int32)
    
    # First pass: count class frequencies
    for _, labels_batch in train_loader:
        for label in labels_batch.numpy():
            client_class_counts[label] += 1
    
    # Compute class weights for balanced loss
    total_samples = sum(client_class_counts)
    class_weights = np.ones(num_classes, dtype=np.float32)
    
    if total_samples > 0:
        for c in range(num_classes):
            if client_class_counts[c] > 0:
                # Inverse frequency weighting (sqrt for moderate scaling)
                freq = client_class_counts[c] / total_samples
                class_weights[c] = np.sqrt(1.0 / (freq + 1e-8))
        
        # Normalize to mean = 1.0
        class_weights = class_weights / (np.mean(class_weights[client_class_counts > 0]) + 1e-8)
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Setup hook for penultimate layer embeddings
    penultimate_embeddings = []
    hook_handle = None
    
    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate_embeddings.append(o)
        )
    else:
        # Fallback: no CRA loss if architecture doesn't support it
        print("Warning: Model architecture incompatible with FedCRA. Using standard training.")
        hook_handle = None
    
    # ══════════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════════
    running_loss = 0.0
    num_steps = 0
    all_labels = []
    all_preds = []
    
    # Reset counts for training tracking
    client_class_counts = np.zeros(num_classes, dtype=np.int32)
    
    # CRA strength gating (warm-up + confidence)
    anchors_ready = int(cra_config.get("cra_anchors_ready", 0))
    cra_enabled = (anchors_ready >= num_classes // 2) and (round_id >= 5)
    avg_confidence = confidence.mean().item()
    cra_strength = min(1.0, round_id / 20.0) * max(0.1, avg_confidence) if cra_enabled else 0.0
    
    try:
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Track class distribution
                for label in labels.cpu().numpy():
                    client_class_counts[label] += 1
                
                # Forward pass
                penultimate_embeddings.clear()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                
                # ──────────────────────────────────────────────────────
                # Loss Component 1: Cross-Entropy
                # ──────────────────────────────────────────────────────
                ce_loss = criterion(outputs, labels)
                
                # ──────────────────────────────────────────────────────
                # Loss Component 2: CRA Alignment Loss
                # ──────────────────────────────────────────────────────
                lambda_cra = cra_config.get("lambda_cra", 0.1)  # Use config value
                cra_loss = torch.tensor(0.0, device=device)
                
                if cra_strength > 0.0 and penultimate_embeddings and anchor_valid.any():
                    embeddings = penultimate_embeddings[0]  # [batch_size, embedding_dim]
                    
                    # L2-normalize embeddings
                    embeddings_norm = embeddings / (
                        embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    )
                    
                    batch_size = labels.size(0)
                    
                    for class_id in range(num_classes):
                        mask = (labels == class_id)
                        
                        # SELECTIVE: Only align classes present in batch
                        if not mask.any():
                            continue
                        
                        # SELECTIVE: Require minimum samples and valid anchor
                        n_samples = mask.sum().item()
                        if n_samples < 3 or not anchor_valid[class_id]:
                            continue
                        
                        # Extract class embeddings
                        class_emb = embeddings_norm[mask]  # [n_samples, emb_dim]
                        anchor = anchors_normalized[class_id]  # [emb_dim]
                        
                        # Squared distance to anchor
                        sq_dist = ((class_emb - anchor) ** 2).mean()
                        
                        # Class frequency weighting
                        freq = n_samples / batch_size
                        class_weight = min(5.0, np.sqrt(1.0 / (freq + 1e-8)))
                        
                        # Confidence weighting with minimum thresholds
                        conf_c = confidence[class_id].item()
                        
                        # Continuous confidence scaling (FIX: prevent zero CRA for low confidence)
                        # Minority classes (freq < 0.2): only skip if conf_c < 0.05, else scale with min 0.2
                        # Majority classes (freq >= 0.2): only skip if conf_c < 0.3, else scale with min 0.4
                        is_minority = freq < 0.2
                        min_conf_threshold = 0.05 if is_minority else 0.3
                        min_scale = 0.2 if is_minority else 0.4
                        
                        if conf_c < min_conf_threshold:
                            conf_weight = 0.0  # Skip entirely
                        else:
                            conf_weight = max(min_scale, conf_c)  # Scale with minimum
                        
                        # Accumulate weighted CRA loss
                        cra_loss = cra_loss + (sq_dist * class_weight * conf_weight)
                    
                    # Apply global CRA strength
                    cra_loss = cra_loss * cra_strength
                
                # ──────────────────────────────────────────────────────
                # Loss Component 3: Class-Conditional Proximal
                # ──────────────────────────────────────────────────────
                proximal_loss = torch.tensor(0.0, device=device)
                
                # Compute effective μ for classes in this batch
                unique_classes = torch.unique(labels)
                if len(unique_classes) > 0:
                    mu_effective = mu_c[unique_classes].mean()
                    
                    # Parameter divergence
                    for param_name, local_param in model.named_parameters():
                        if param_name in global_params_dict:
                            global_param = global_params_dict[param_name]
                            diff = local_param - global_param
                            proximal_loss = proximal_loss + (diff ** 2).sum()
                    
                    proximal_loss = (mu_effective / 2.0) * proximal_loss
                
                # ──────────────────────────────────────────────────────
                # Combined Loss
                # ──────────────────────────────────────────────────────
                total_loss = ce_loss + lambda_cra * cra_loss + proximal_loss
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                
                # Logging
                running_loss += total_loss.item()
                num_steps += 1
                
                preds = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
    
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    # ══════════════════════════════════════════════════════════════════
    # Extract Residuals for Server Aggregation
    # ══════════════════════════════════════════════════════════════════
    residuals = _extract_class_residuals(
        model, train_loader, num_classes, anchors.cpu().numpy(), device
    )
    
    # ══════════════════════════════════════════════════════════════════
    # Compute Metrics
    # ══════════════════════════════════════════════════════════════════
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Confusion matrix for FPR
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    FPR = FP / (FP + TN + 1e-10)
    
    metrics = {
        "loss": running_loss / max(num_steps, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "error_rate": 1.0 - accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_fpr": float(np.mean(FPR)),
        "cra_residuals": json.dumps(residuals),
        "cra_class_counts": json.dumps(
            {int(k): int(v) for k, v in enumerate(client_class_counts)}
        ),
    }
    
    return metrics


def _extract_class_residuals(
    model: nn.Module,
    loader,
    num_classes: int,
    anchors: np.ndarray,
    device: torch.device,
) -> Dict[int, list]:
    """
    Extract per-class mean embeddings (residuals) for server aggregation.
    
    Returns dict mapping class_id → mean embedding vector (as list).
    """
    model.eval()
    
    # Setup hook for penultimate layer
    penultimate = []
    hook_handle = None
    
    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate.append(o.detach())
        )
    else:
        return {}
    
    # Accumulate embeddings per class
    class_embeddings_sum = {}
    class_counts = {}
    
    try:
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels_np = labels.numpy()
                
                penultimate.clear()
                _ = model(inputs)
                
                if not penultimate:
                    continue
                
                embeddings = penultimate[0].cpu().numpy()  # [batch_size, emb_dim]
                
                # L2-normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms < 1e-8, 1.0, norms)
                embeddings = embeddings / norms
                
                # Accumulate per class
                for class_id in range(num_classes):
                    mask = (labels_np == class_id)
                    if mask.sum() == 0:
                        continue
                    
                    class_emb = embeddings[mask]
                    
                    if class_id not in class_embeddings_sum:
                        class_embeddings_sum[class_id] = np.zeros(
                            class_emb.shape[1], dtype=np.float32
                        )
                        class_counts[class_id] = 0
                    
                    class_embeddings_sum[class_id] += class_emb.sum(axis=0)
                    class_counts[class_id] += int(mask.sum())
    
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    # Compute mean embeddings (residuals)
    residuals = {}
    for class_id in class_embeddings_sum:
        if class_counts[class_id] > 0:
            mean_embedding = class_embeddings_sum[class_id] / class_counts[class_id]
            residuals[class_id] = mean_embedding.tolist()
    
    return residuals