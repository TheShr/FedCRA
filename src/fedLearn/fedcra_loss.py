"""
FedCRA Loss Implementation — Presentation-Aligned
=================================================

Based on FedCRA presentation specifications:

OBJECTIVE:
   min Σ_c [ F_c(θ_c) + μ_c/2 ‖θ_c − θ_{g,c}‖² ]

Where:
- F_c: Per-class loss component
- μ_c: Adaptive class-specific penalty (from server)
- θ_c: Client parameters for class c
- θ_{g,c}: Global anchor parameters for class c

KEY DESIGN PRINCIPLES:
1. Selective Alignment: Only apply CRA loss for classes present in client data
2. Confidence Scaling: Scale by anchor confidence (conf_c)
3. Class Weighting: Balance minority/majority classes
4. Warm-up Period: Gate CRA loss early in training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from typing import Dict, Tuple, Optional


class CRALoss(nn.Module):
    """
    Class-Residual Anchoring (CRA) Loss for FedCRA.
    
    This implements the anchor alignment term in the FedCRA objective.
    The proximal term (μ_c regularization) is applied separately in training.
    
    Design:
    - Pulls class embeddings toward their global anchors
    - Uses confidence-based scaling to handle noisy anchors
    - Applies selective alignment (skip absent classes)
    - Includes warm-up to avoid early-round noise
    """
    
    def __init__(self, cra_config: dict):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parse configuration
        self.anchors = torch.tensor(
            json.loads(cra_config["cra_anchors"]),
            dtype=torch.float32,
            device=self.device
        )  # [num_classes, embedding_dim]
        
        self.confidence = torch.tensor(
            json.loads(cra_config["cra_confidence"]),
            dtype=torch.float32,
            device=self.device
        )  # [num_classes]
        
        self.num_classes = int(cra_config["cra_num_classes"])
        self.proximal_mu = float(cra_config.get("cra_proximal_mu", 0.01))
        self.round_id = int(cra_config.get("round_id", 0))
        self.anchors_ready = int(cra_config.get("cra_anchors_ready", 0))
        self.use_anchor_alignment = bool(cra_config.get("cra_use_anchor_alignment", True))
        self.use_class_penalty = bool(cra_config.get("cra_use_class_penalty", True))
        
        # Parse class-conditional penalties (μ_c)
        if "cra_mu_c" in cra_config:
            self.class_mu = torch.tensor(
                json.loads(cra_config["cra_mu_c"]),
                dtype=torch.float32,
                device=self.device
            )  # [num_classes]
        else:
            self.class_mu = None
        
        # Compute CRA strength (gating mechanism)
        self.cra_strength = self._compute_cra_strength()
    
    def _compute_cra_strength(self) -> float:
        """
        Adaptive CRA strength with three gates:
        
        1. Anchor Readiness: Need at least half anchors initialized
        2. Confidence: Scale by average anchor confidence
        3. Warm-up: Ramp from 0→1 over first 15 rounds
        
        Returns strength factor ∈ [0, 1]
        """
        # Gate 1: Anchor readiness
        if self.anchors_ready <= 0:
            return 0.0  # No anchors yet → skip CRA

        if not self.use_anchor_alignment:
            return 0.0

        # Gate 1.5: partial readiness scaling for imbalanced classes
        readiness = min(1.0, self.anchors_ready / max(1, self.num_classes / 3))
        
        # Gate 2: Confidence scaling (more aggressive, less penalty)
        avg_confidence = self.confidence.mean().item()
        conf_scale = float(np.clip(avg_confidence, 0.3, 1.0))
        
        # Lighter penalty for low confidence
        if avg_confidence < 0.3:
            conf_scale *= 0.85
        
        # Gate 3: Warm-up schedule
        # Start CRA earlier for extreme heterogeneity to help anchor initialization.
        if self.round_id < 2:
            return 0.0  # Skip CRA entirely for first round only
        warmup_factor = min(1.0, self.round_id / 12.0)
        return warmup_factor * conf_scale * readiness
    
    def compute_class_residuals(
        self,
        embeddings: torch.Tensor,  # [batch_size, embedding_dim]
        labels: torch.Tensor,       # [batch_size]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """
        Compute per-class mean embeddings (residuals) for server aggregation.
        
        Returns:
        --------
        mean_embeddings : dict[int, np.ndarray]
            Per-class mean embedding vector [embedding_dim]
            Used by server to update global anchors
            
        class_counts : dict[int, int]
            Number of samples per class in this batch
        """
        mean_embeddings: Dict[int, np.ndarray] = {}
        class_counts: Dict[int, int] = {}
        
        for class_id in range(self.num_classes):
            mask = (labels == class_id)
            
            if mask.any():
                class_embeddings = embeddings[mask]  # [n_samples, embedding_dim]
                
                # Compute mean embedding for this class
                mean_embedding = class_embeddings.detach().cpu().mean(dim=0).numpy()
                mean_embeddings[class_id] = mean_embedding
                
                # Count samples
                class_counts[class_id] = int(mask.sum().item())
        
        return mean_embeddings, class_counts
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [batch_size, embedding_dim] — L2-normalized
        labels: torch.Tensor,       # [batch_size]
    ) -> torch.Tensor:
        """
        Compute CRA alignment loss.
        
        For each class c present in the batch:
           loss_c = conf_c × class_weight_c × ||embedding_c - anchor_c||²
        
        Where:
        - conf_c: Anchor confidence (from server)
        - class_weight_c: Inverse frequency weighting (balance classes)
        - embedding_c: Mean embedding for class c in batch
        - anchor_c: Global anchor for class c
        
        Parameters:
        -----------
        embeddings : torch.Tensor [batch_size, embedding_dim]
            L2-normalized embeddings from penultimate layer
            
        labels : torch.Tensor [batch_size]
            Class labels for each sample
        
        Returns:
        --------
        cra_loss : torch.Tensor (scalar)
            Weighted CRA alignment loss
        """
        # Early exit if CRA is disabled
        if self.cra_strength <= 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        
        batch_size = labels.size(0)
        total_loss = torch.tensor(0.0, device=self.device)
        
        for class_id in range(self.num_classes):
            mask = (labels == class_id)
            
            # Selective alignment: skip absent classes
            if not mask.any():
                continue
            
            # Extract embeddings for this class
            class_embeddings = embeddings[mask]  # [n_class, embedding_dim]
            n_samples = mask.sum().item()
            
            # Get global anchor for this class
            anchor = self.anchors[class_id]  # [embedding_dim]
            
            # Skip invalid/uninitialised anchors
            if anchor.norm() < 1e-6:
                continue
            
            # Compute squared distance to anchor
            squared_dist = ((class_embeddings - anchor) ** 2).mean()
            
            # Class weighting: balance minority/majority classes
            # Higher cap for extreme heterogeneity to prioritize rare samples.
            class_frequency = n_samples / batch_size
            class_weight = min(4.5, math.sqrt(1.0 / (class_frequency + 1e-8)))
            
            # Confidence weighting with aggressive minimums for minority classes.
            conf_c = self.confidence[class_id].item()
            is_minority = class_frequency < 0.2
            min_conf_threshold = 0.05 if is_minority else 0.25
            min_scale = 0.5 if is_minority else 0.5
            
            if conf_c < min_conf_threshold:
                confidence_weight = 0.0  # Skip entirely
            else:
                # Higher minimum scaling for both minority and majority in extreme heterogeneity
                confidence_weight = min(1.2, max(min_scale, conf_c))
            
            # Accumulate weighted loss
            total_loss = total_loss + (
                squared_dist * class_weight * confidence_weight
            )
        
        # Apply global CRA strength
        final_loss = total_loss * self.cra_strength
        
        return final_loss


def compute_proximal_loss(
    model: nn.Module,
    global_params: Dict[str, torch.Tensor],  # Global parameters
    mu_c: torch.Tensor,                       # [num_classes] — adaptive penalties
    labels: torch.Tensor,                     # [batch_size]
    device: torch.device,
) -> torch.Tensor:
    """
    Class-conditional proximal regularization.
    
    This is the core FedCRA innovation over FedProx:
       FedProx: μ/2 × ||θ - θ_g||²  (single global μ)
       FedCRA:  μ_c/2 × ||θ - θ_g||²  (per-class adaptive μ_c)
    
    The effective μ is computed as the average of μ_c for classes present
    in the current batch. This provides class-conditional regularization
    while keeping the parameter-space penalty simple.
    
    Parameters:
    -----------
    model : nn.Module
        Current local model
        
    global_params : dict[str, torch.Tensor]
        Global model parameters (from server)
        
    mu_c : torch.Tensor [num_classes]
        Per-class adaptive penalties (from server)
        
    labels : torch.Tensor [batch_size]
        Class labels in current batch
        
    device : torch.device
        Device for computation
    
    Returns:
    --------
    proximal_loss : torch.Tensor (scalar)
        Class-conditional proximal regularization term
    """
    # Compute effective μ: average over classes in this batch
    unique_classes = torch.unique(labels)
    
    if len(unique_classes) == 0:
        return torch.tensor(0.0, device=device)
    
    # Average μ_c for present classes
    mu_effective = mu_c[unique_classes].mean()
    
    # Compute parameter divergence
    proximal_term = torch.tensor(0.0, device=device)
    
    for param_name, local_param in model.named_parameters():
        if param_name in global_params:
            global_param = global_params[param_name].to(device)
            diff = local_param - global_param
            proximal_term = proximal_term + (diff ** 2).sum()
    
    # Apply class-conditional penalty
    proximal_loss = (mu_effective / 2.0) * proximal_term
    
    return proximal_loss


class FedCRATrainer:
    """
    Helper class for FedCRA training workflow.
    
    Encapsulates the complete training loop with:
    1. Cross-entropy loss (with optional class weighting)
    2. CRA alignment loss (embedding-space)
    3. Proximal regularization (parameter-space)
    """
    
    def __init__(
        self,
        model: nn.Module,
        cra_config: dict,
        global_params: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize CRA loss
        self.cra_loss_fn = CRALoss(cra_config)
        
        # Store global parameters for proximal term
        self.global_params = {
            name: param.clone().detach()
            for name, param in global_params.items()
        }
        
        # Parse μ_c from config
        if "cra_mu_c" in cra_config:
            self.mu_c = torch.tensor(
                json.loads(cra_config["cra_mu_c"]),
                dtype=torch.float32,
                device=self.device
            )
        else:
            # Fallback to uniform μ
            num_classes = int(cra_config["cra_num_classes"])
            proximal_mu = float(cra_config.get("cra_proximal_mu", 0.01))
            self.mu_c = torch.full((num_classes,), proximal_mu, device=self.device)
        
        # Training configuration
        self.num_classes = int(cra_config["cra_num_classes"])
        self.round_id = int(cra_config.get("round_id", 0))
        
        # Coefficients
        self.cra_coefficient = float(cra_config.get("lambda_cra", 0.1))  # Use config value
        self.proximal_coefficient = 1.0
    
    def compute_loss(
        self,
        outputs: torch.Tensor,      # [batch_size, num_classes]
        embeddings: torch.Tensor,   # [batch_size, embedding_dim]
        labels: torch.Tensor,        # [batch_size]
        criterion: nn.Module,        # Cross-entropy loss
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined FedCRA loss.
        
        Total Loss = CE + λ_CRA × CRA + Proximal
        
        Returns:
        --------
        total_loss : torch.Tensor
            Combined loss for backpropagation
            
        loss_components : dict
            Breakdown of loss components for logging
        """
        # 1. Cross-entropy loss
        ce_loss = criterion(outputs, labels)
        
        # 2. CRA alignment loss
        cra_loss = self.cra_loss_fn(embeddings, labels)
        
        # 3. Proximal regularization
        proximal_loss = compute_proximal_loss(
            self.model,
            self.global_params,
            self.mu_c,
            labels,
            self.device
        )
        
        # 4. Combine losses
        total_loss = (
            ce_loss + 
            self.cra_coefficient * cra_loss +
            self.proximal_coefficient * proximal_loss
        )
        
        # Prepare logging dict
        loss_components = {
            "total": total_loss.item(),
            "ce": ce_loss.item(),
            "cra": cra_loss.item(),
            "proximal": proximal_loss.item(),
        }
        
        return total_loss, loss_components
    
    def extract_residuals(
        self,
        train_loader,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """
        Extract per-class residuals for server aggregation.
        
        Returns mean embeddings and class counts.
        """
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        # Hook to extract penultimate layer
        penultimate = []
        hook_handle = None
        
        if hasattr(self.model, "fc_layers") and len(self.model.fc_layers) >= 2:
            hook_handle = self.model.fc_layers[-2].register_forward_hook(
                lambda m, i, o: penultimate.append(o.detach())
            )
        else:
            return {}, {}
        
        try:
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    penultimate.clear()
                    _ = self.model(inputs)
                    
                    if penultimate:
                        embeddings = penultimate[0]
                        
                        # L2-normalize embeddings
                        embeddings = embeddings / (
                            embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        )
                        
                        all_embeddings.append(embeddings.cpu())
                        all_labels.append(labels.cpu())
        finally:
            if hook_handle:
                hook_handle.remove()
        
        if not all_embeddings:
            return {}, {}
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute per-class means
        return self.cra_loss_fn.compute_class_residuals(all_embeddings, all_labels)