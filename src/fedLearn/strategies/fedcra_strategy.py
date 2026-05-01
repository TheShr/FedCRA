"""
FedCRA Strategy — Revised Implementation
==========================================

Based on FedCRA Presentation Specifications:

FOUR NOVEL COMPONENTS:
A. Class-Conditional Reliability Weighting
   - r_kc = samples(c,k) / total(k)
   - Aggregate weight ∝ reliability_kc

B. Distribution-Aware Client Selection
   - γ_k = 1 − H(label dist) / log(K)
   - Specialist clients prioritized

C. Anchor Confidence Scaling
   - conf_c = 1/var(residuals_c)
   - μ_c = μ_base × conf_c

D. Selective Class Alignment
   - Skip anchor alignment for classes not present at client
   - Natural handling of label shift

THEORETICAL CONVERGENCE:
   O(1/T + Σ_c σ_c² / (μ_c T))
   
Per-class adaptive penalties ensure better convergence than FedProx's global μ.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from scipy.stats import entropy


class FedCRA(FedAvg):
    """
    FedCRA: Class-Conditional Regularization for Federated Learning
    
    Addresses label heterogeneity through four novel mechanisms:
    1. Class-conditional reliability weighting for aggregation
    2. Distribution-aware client selection via entropy
    3. Anchor confidence scaling for adaptive penalties
    4. Selective class alignment (skip absent classes)
    """
    
    def __init__(
        self,
        config,
        *,
        server_metrics_dir: Optional[str] = None,
        save_anchor_logs: bool = False,
        server_save: Optional[Callable] = None,
        **kwargs,
    ):
        # Filter kwargs to only include valid FedAvg parameters
        valid_kwargs = {
            "fraction_fit", "fraction_evaluate", "min_fit_clients",
            "min_evaluate_clients", "min_available_clients", "evaluate_fn",
            "on_fit_config_fn", "on_evaluate_config_fn", "accept_failures",
            "initial_parameters", "fit_metrics_aggregation_fn",
            "evaluate_metrics_aggregation_fn",
        }
        super().__init__(**{k: v for k, v in kwargs.items() if k in valid_kwargs})
        
        # FedCRA configuration
        self.config = config
        self.server_metrics_dir = server_metrics_dir
        self.save_anchor_logs = save_anchor_logs
        self._server_save = server_save
        
        # Base parameters
        self.proximal_mu = config.strategy.params.proximal_mu
        self.embedding_dim = config.strategy.params.embedding_dim
        self.num_classes = config.strategy.params.num_classes
        self.base_lr = config.config_fit.learning_rate
        
        # FedCRA-specific parameters
        self.lambda_cra = config.strategy.params.get("lambda_cra", 0.1)
        self.use_class_penalty = config.strategy.params.get("use_class_penalty", True)
        self.use_anchor_alignment = config.strategy.params.get("use_anchor_alignment", True)
        
        # Adaptive lambda_cra schedule (curriculum learning)
        self.lambda_cra_initial = config.strategy.params.get("lambda_cra_initial", 0.10)
        self.lambda_cra_medium = config.strategy.params.get("lambda_cra_medium", 0.18)
        self.lambda_cra_base = config.strategy.params.get("lambda_cra_base", self.lambda_cra)
        self.anchor_ready_threshold = config.strategy.params.get("anchor_ready_threshold", max(1, self.num_classes // 4))
        
        # Anchor storage (Component C: Anchor Confidence Scaling)
        self.global_anchors = np.zeros((self.num_classes, self.embedding_dim), dtype=np.float32)
        self.anchor_confidence = np.ones(self.num_classes, dtype=np.float32) * 0.1
        self.anchor_initialized = np.zeros(self.num_classes, dtype=bool)
        self.anchors_ready = 0
        
        # Global statistics
        self.global_class_counts = np.zeros(self.num_classes, dtype=np.float32)
        self.client_weights: Dict[str, float] = {}
        self.client_entropy: Dict[str, float] = {}
        
        # Per-class adaptive penalties (μ_c)
        self._class_conditional_penalties = np.full(
            self.num_classes, self.proximal_mu, dtype=np.float32
        )
        
        # Logging
        self._cra_log: List[Dict] = []
        
    # ═══════════════════════════════════════════════════════════════════════
    # COMPONENT A: Class-Conditional Reliability Weighting
    # ═══════════════════════════════════════════════════════════════════════
    
    def _compute_reliability_weights(
        self, 
        client_class_counts: Dict[str, Dict[int, int]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute r_kc = samples(c,k) / total(k)
        
        Returns per-client, per-class reliability weights.
        Higher weight for classes that are well-represented at that client.
        """
        reliability_weights = {}
        
        for client_id, counts in client_class_counts.items():
            total_samples = sum(counts.values())
            if total_samples == 0:
                continue
                
            r_kc = np.zeros(self.num_classes, dtype=np.float32)
            for class_id, count in counts.items():
                r_kc[class_id] = count / (total_samples + 1e-9)
                
            reliability_weights[client_id] = r_kc
            
        return reliability_weights
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPONENT B: Distribution-Aware Client Selection
    # ═══════════════════════════════════════════════════════════════════════
    
    def _compute_client_selection_scores(
        self,
        client_class_counts: Dict[str, Dict[int, int]]
    ) -> Dict[str, float]:
        """
        Compute γ_k = 1 − H(label dist) / log(K)
        
        Specialist clients (low entropy) get higher scores.
        Generalist clients (high entropy) get lower scores.
        
        This prioritizes rare-class specialists during client selection.
        """
        selection_scores = {}
        
        for client_id, counts in client_class_counts.items():
            total_samples = sum(counts.values())
            if total_samples == 0:
                selection_scores[client_id] = 0.0
                continue
            
            # Compute label distribution
            probs = np.array([
                counts.get(c, 0) / total_samples 
                for c in range(self.num_classes)
            ])
            probs = probs[probs > 0]  # Only non-zero classes
            
            if len(probs) == 0:
                selection_scores[client_id] = 0.0
                continue
            
            # Entropy calculation
            H = entropy(probs, base=2)
            max_entropy = math.log2(self.num_classes)
            
            # γ_k: specialists (low H) → high score
            gamma_k = 1.0 - (H / (max_entropy + 1e-9))
            
            selection_scores[client_id] = gamma_k
            
        return selection_scores
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPONENT C: Anchor Confidence Scaling
    # ═══════════════════════════════════════════════════════════════════════
    
    def _update_anchor_confidence(
        self,
        client_residuals: Dict[str, Dict[int, np.ndarray]]
    ):
        """
        conf_c = 1 / var(residuals_c)
        
        Compute confidence based on inter-client agreement.
        Low variance → high confidence → anchor is reliable.
        High variance → low confidence → anchor is unreliable.
        """
        for class_id in range(self.num_classes):
            # Collect residuals for this class from all clients
            class_residuals = [
                residuals[class_id] 
                for residuals in client_residuals.values() 
                if class_id in residuals
            ]
            
            if len(class_residuals) >= 2:
                # Stack and compute variance
                residual_matrix = np.stack(class_residuals)  # [n_clients, emb_dim]
                variance = np.var(residual_matrix, axis=0).mean()
                
                # Confidence = 1 / (variance + epsilon), clamped to conservative range for stability.
                # Higher epsilon to reduce sensitivity to small variance changes.
                raw_confidence = 1.0 / (variance + 1e-3)
                self.anchor_confidence[class_id] = np.clip(raw_confidence, 0.05, 0.8)
            else:
                # Insufficient data → low confidence
                self.anchor_confidence[class_id] = 0.05
    
    def _update_anchors_with_reliability(
        self,
        client_residuals: Dict[str, Dict[int, np.ndarray]],
        reliability_weights: Dict[str, np.ndarray]
    ):
        """
        Update global anchors using reliability-weighted aggregation.
        
        Uses EMA with confidence-adaptive momentum:
        - High confidence → slow update (preserve good anchor)
        - Low confidence → fast update (escape bad anchor)
        """
        median_conf = np.median(self.anchor_confidence) + 1e-9
        
        for class_id in range(self.num_classes):
            weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
            total_weight = 0.0
            
            # Aggregate residuals weighted by reliability
            for client_id, residuals in client_residuals.items():
                if class_id not in residuals:
                    continue
                    
                if client_id not in reliability_weights:
                    continue
                
                r_kc = reliability_weights[client_id][class_id]
                if r_kc <= 0:
                    continue
                
                weighted_sum += r_kc * residuals[class_id]
                total_weight += r_kc
            
            if total_weight <= 0:
                continue
            
            # Compute new anchor (normalized)
            new_anchor = weighted_sum / total_weight
            norm = np.linalg.norm(new_anchor)
            if norm > 1e-9:
                new_anchor = new_anchor / norm
            
            # Initialize or update with confidence-adaptive EMA
            if not self.anchor_initialized[class_id]:
                self.global_anchors[class_id] = new_anchor
                self.anchor_initialized[class_id] = True
            else:
                # Confidence-adaptive momentum; keep anchors adaptive enough for noisy or rare classes.
                conf_ratio = float(self.anchor_confidence[class_id]) / median_conf
                momentum = float(np.clip(0.65 + 0.2 * conf_ratio, 0.65, 0.9))
                
                self.global_anchors[class_id] = (
                    momentum * self.global_anchors[class_id] + 
                    (1 - momentum) * new_anchor
                )
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPONENT D: Selective Class Alignment (via μ_c computation)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _compute_class_conditional_penalties(
        self,
        client_class_counts: Dict[str, Dict[int, int]]
    ) -> np.ndarray:
        """
        Compute μ_c = μ_base × f(reliability_c, entropy_c, drift_c)
        
        Per presentation slide:
        - Reliability: # samples of class c / total
        - Entropy: Client label distribution balance
        - Drift: Per-class residual variance (captured in confidence)
        - Confidence: 1 / variance of residuals
        
        Design:
        - Rare classes (low reliability) → HIGH μ_c (need more regularization)
        - Common classes (high reliability) → LOW μ_c (natural signal sufficient)
        - High confidence → scale UP μ_c (anchor is reliable)
        - Low confidence → scale DOWN μ_c (anchor may be noisy)
        """
        if not self.use_class_penalty:
            return np.full(self.num_classes, self.proximal_mu, dtype=np.float32)

        mu_c = np.full(self.num_classes, self.proximal_mu, dtype=np.float32)
        
        # Compute global statistics
        total_samples = sum(
            sum(counts.values()) 
            for counts in client_class_counts.values()
        ) + 1e-9
        
        class_samples = np.zeros(self.num_classes, dtype=np.float32)
        class_entropy = np.zeros(self.num_classes, dtype=np.float32)
        n_clients = len(client_class_counts)
        
        # Per-class statistics
        for class_id in range(self.num_classes):
            # Total samples for this class across all clients
            samples_c = sum(
                counts.get(class_id, 0) 
                for counts in client_class_counts.values()
            )
            class_samples[class_id] = samples_c
            
            if samples_c > 0:
                # Compute entropy of class distribution across clients
                client_probs = np.array([
                    counts.get(class_id, 0) / samples_c
                    for counts in client_class_counts.values()
                    if counts.get(class_id, 0) > 0
                ])
                
                if len(client_probs) > 0:
                    class_entropy[class_id] = entropy(client_probs, base=2)
        
        # Normalize entropy
        max_entropy = math.log2(n_clients) if n_clients > 0 else 1.0
        normalized_entropy = class_entropy / (max_entropy + 1e-9)
        
        # Global reliability (fraction of total samples)
        reliability = class_samples / total_samples
        
        # Compute adaptive μ_c for each class
        for class_id in range(self.num_classes):
            if reliability[class_id] > 0:
                # Factor 1: Inverse reliability (rare classes need stronger pull)
                # Use sqrt to moderate the effect
                inv_reliability = 1.0 / (math.sqrt(float(reliability[class_id])) + 1e-9)
                
                # Factor 2: Entropy factor (concentrated distribution is better)
                # High entropy → data is spread out → need more regularization
                entropy_factor = 1.0 + float(normalized_entropy[class_id])
                
                # Factor 3: Confidence scaling (from Component C)
                # High confidence → reliable anchor → can use stronger penalty
                # Normalize by median confidence
                median_conf = np.median(self.anchor_confidence) + 1e-9
                conf_scale = float(self.anchor_confidence[class_id]) / median_conf
                conf_scale = np.clip(conf_scale, 0.5, 2.0)  # Moderate scaling
                
                # Combine factors
                mu_c[class_id] = (
                    self.proximal_mu * 
                    inv_reliability * 
                    entropy_factor * 
                    conf_scale
                )
                
                # Clamp to a wider range [0.5μ, 4μ] so rare classes can receive
                # stronger class-conditional proximal regularization than FedProx.
                mu_c[class_id] = float(np.clip(
                    mu_c[class_id],
                    self.proximal_mu * 0.5,
                    self.proximal_mu * 4.0
                ))
            else:
                # Unseen class → conservative regularization
                mu_c[class_id] = self.proximal_mu * 4.0
        
        return mu_c
    
    # ═══════════════════════════════════════════════════════════════════════
    # Client Weighting for Aggregation
    # ═══════════════════════════════════════════════════════════════════════
    
    def _compute_client_weights(
        self,
        client_class_counts: Dict[str, Dict[int, int]]
    ):
        """
        Compute client weights for aggregation combining:
        1. Entropy-based selection score (Component B)
        2. Data size scaling (with smoothing)
        """
        total_global_samples = sum(
            sum(counts.values()) 
            for counts in client_class_counts.values()
        ) + 1e-9
        
        # Get entropy-based selection scores
        selection_scores = self._compute_client_selection_scores(client_class_counts)
        
        for client_id, counts in client_class_counts.items():
            total_samples = sum(counts.values())
            
            if total_samples == 0:
                self.client_weights[client_id] = 0.0
                continue
            
            # Selection score (specialist bonus)
            gamma_k = selection_scores.get(client_id, 0.5)
            
            # Baseline proportional weight (FedAvg-like) with a safe specialist bonus.
            baseline_weight = total_samples / total_global_samples
            specialist_bonus = 0.5 + 0.5 * gamma_k  # range [0.5, 1.0]
            
            self.client_weights[client_id] = baseline_weight * specialist_bonus
            
            # Store entropy for logging
            total = sum(counts.values())
            probs = np.array([counts.get(c, 0) / total for c in range(self.num_classes)])
            probs = probs[probs > 0]
            self.client_entropy[client_id] = float(entropy(probs, base=2) if len(probs) > 0 else 0.0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Server-side Flower Integration
    # ═══════════════════════════════════════════════════════════════════════
    
    def _get_adaptive_lambda_cra(self, server_round: int) -> float:
        """
        Curriculum learning schedule for λ_cra to prevent anchor collapse.
        
        Strategy:
        - Rounds 1-5:   λ = lambda_cra_initial (weak)   — Let model learn basic features
        - Rounds 6-15:  λ = lambda_cra_medium (medium)  — Anchors stabilize, gradual increase
        - Rounds 16+:   λ = lambda_cra_base (full)      — Mature anchors, aggressive alignment
        
        Problem it solves:
        At α=0.1 (extreme heterogeneity), high λ_cra in Round 1 pulls embeddings toward
        noisy anchors → model collapses to 2-3 classes. This schedule allows anchors
        to initialize with good data before applying strong CRA penalties.
        
        Configurable via:
        - ++strategy.params.lambda_cra_initial=0.10
        - ++strategy.params.lambda_cra_medium=0.18
        - ++strategy.params.lambda_cra_base=0.32
        """
        if server_round <= 5:
            base_lambda = self.lambda_cra_initial
        elif server_round <= 15:
            base_lambda = self.lambda_cra_medium
        else:
            base_lambda = self.lambda_cra_base

        readiness = min(1.0, self.anchors_ready / max(1, self.anchor_ready_threshold))
        return base_lambda * (0.6 + 0.4 * readiness)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ):
        """Send global model + FedCRA config to selected clients."""
        configs = super().configure_fit(server_round, parameters, client_manager)
        
        if not configs:
            return configs
        
        # Get adaptive lambda_cra based on server round
        adaptive_lambda_cra = self._get_adaptive_lambda_cra(server_round)
        
        # Serialize FedCRA state for clients
        anchors_json = json.dumps(self.global_anchors.tolist())
        confidence_json = json.dumps(self.anchor_confidence.tolist())
        anchors_ready = int(self.anchor_initialized.sum())
        mu_c_json = json.dumps(self._class_conditional_penalties.tolist())
        
        # Patch configurations
        patched_configs = []
        for client_proxy, fit_ins in configs:
            config = dict(fit_ins.config)
            
            # FedCRA configuration
            config["cra_anchors"] = anchors_json
            config["cra_confidence"] = confidence_json
            config["cra_anchors_ready"] = anchors_ready
            config["cra_proximal_mu"] = self.proximal_mu
            config["cra_num_classes"] = self.num_classes
            config["cra_mu_c"] = mu_c_json
            config["cra_use_class_penalty"] = self.use_class_penalty
            config["cra_use_anchor_alignment"] = self.use_anchor_alignment
            config["lambda_cra"] = adaptive_lambda_cra  # ← Use adaptive schedule
            config["learning_rate"] = self.base_lr
            config["round_id"] = server_round
            
            patched_configs.append((
                client_proxy,
                fl.common.FitIns(parameters=fit_ins.parameters, config=config)
            ))
        
        return patched_configs
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures
    ):
        """
        Aggregate client updates using:
        1. Reliability-weighted anchor updates (Component A)
        2. Distribution-aware client weights (Component B)
        3. Confidence-based scaling (Component C)
        4. Selective alignment via μ_c (Component D)
        """
        if not results:
            return None, {}
        
        # Extract client data
        client_residuals: Dict[str, Dict[int, np.ndarray]] = {}
        client_class_counts: Dict[str, Dict[int, int]] = {}
        
        for idx, (client_proxy, fit_res) in enumerate(results):
            metrics = fit_res.metrics or {}
            
            # Get client ID from metrics (more reliable than proxy)
            client_id = str(metrics.get("client_id", idx))
            
            # Extract residuals and class counts
            residuals_json = metrics.get("cra_residuals")
            counts_json = metrics.get("cra_class_counts")
            
            if residuals_json and counts_json:
                try:
                    client_residuals[client_id] = {
                        int(k): np.array(v, dtype=np.float32)
                        for k, v in json.loads(residuals_json).items()
                    }
                    client_class_counts[client_id] = {
                        int(k): int(v)
                        for k, v in json.loads(counts_json).items()
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    pass  # Skip malformed data
        
        # Update FedCRA components
        if client_residuals:
            # Component A: Compute reliability weights
            reliability_weights = self._compute_reliability_weights(client_class_counts)
            
            # Component C: Update anchor confidence
            self._update_anchor_confidence(client_residuals)
            
            # Component A + C: Update anchors with reliability weighting
            self._update_anchors_with_reliability(client_residuals, reliability_weights)
        
        if client_class_counts:
            # Component B: Compute client weights (entropy-based)
            self._compute_client_weights(client_class_counts)
            
            # Component D: Compute class-conditional penalties (μ_c)
            if self.use_class_penalty:
                self._class_conditional_penalties = (
                    self._compute_class_conditional_penalties(client_class_counts)
                )
        
        # Aggregate model parameters using distribution-aware weights
        total_weight = sum(self.client_weights.values())
        
        if self.client_weights and total_weight > 0:
            # Normalize weights
            normalized_weights = {
                cid: w / total_weight 
                for cid, w in self.client_weights.items()
            }
            
            # Extract parameter arrays
            param_arrays = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            
            # Weighted aggregation layer by layer
            n_layers = len(param_arrays[0])
            aggregated_params = []
            
            for layer_idx in range(n_layers):
                weighted_layer = None
                
                for idx, (_, fit_res) in enumerate(results):
                    metrics = fit_res.metrics or {}
                    client_id = str(metrics.get("client_id", idx))
                    
                    weight = normalized_weights.get(client_id, 1.0 / len(results))
                    layer_params = param_arrays[idx][layer_idx]
                    
                    if weighted_layer is None:
                        weighted_layer = weight * layer_params
                    else:
                        weighted_layer += weight * layer_params
                
                aggregated_params.append(weighted_layer)
            
            aggregated_parameters = ndarrays_to_parameters(aggregated_params)
        else:
            # Fallback to FedAvg if no weights computed
            aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        
        # Save model if callback provided
        if self._server_save:
            self._server_save(server_round, aggregated_parameters)
        
        # Log FedCRA state
        self._cra_log.append({
            "round": server_round,
            "anchor_confidence": self.anchor_confidence.tolist(),
            "anchors_initialized": int(self.anchor_initialized.sum()),
            "client_weights": dict(self.client_weights),
            "client_entropy": dict(self.client_entropy),
            "mu_c": self._class_conditional_penalties.tolist(),
        })
        
        # Save logs if enabled
        if self.server_metrics_dir and self.save_anchor_logs:
            Path(self.server_metrics_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(self.server_metrics_dir) / "cra_anchor_log.json"
            log_path.write_text(json.dumps(self._cra_log, indent=2))
        
        return aggregated_parameters, {}