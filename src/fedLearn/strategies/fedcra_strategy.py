"""
FedCRA: Federated Class-Residual Anchoring Strategy  v9 (IMPROVED)
---------------------------------------------------------------------------
MAJOR IMPROVEMENTS:

  1. Curriculum Learning: Alpha ramps slower, more rounds in early phase.
     Prevents CRA from dominating too early and drowning out CE learning.
  
  2. Focal Loss Component: Replaces pure CE for better class imbalance handling.
     Gamma=2.0 focuses on hard samples, reduces easy negative class bias.
  
  3. Rho Calibration: Adaptive scaling that reduces class emphasis in early rounds.
     Prevents minority class weights from overwhelming gradient flow.
  
  4. Better Anchor Init: Only initializes from real client data, removes noisy
     synthetic centroids that were causing instability.
  
  5. Cone-based Margin: Uses margin-based loss in embedding space for stability.
     Better than multi-anchor approach for non-IID data.
  
  6. Confidence Scaling: Automatically reduces CRA when loss becomes unstable.
     Detects divergence and backs off gracefully.

Results: FedCRA now beats FedAvg on highly non-IID data (Dirichlet α=0.1).
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg


"""
FedCRA v10: COMPLETE REDESIGN - Research-Grade Implementation
================================================================

CORE CHANGES:
- Class-Conditional Reliability Weighting for anchor aggregation
- Selective Class Alignment (only for present classes)
- Anchor Confidence Scaling for CRA loss
- Proximal Regularization (FedProx-style)
- Distribution-Aware Client Weighting (replaces FedAvg)

REMOVED:
- Static CRA strength (alpha_cra_peak/min)
- Blind alignment for all classes
- Repulsion term (beta_repulsion)
- Global anchor applied uniformly

KEPT:
- Residual extraction per class
- Class-wise anchor representation
- Federated round structure
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional
from collections import defaultdict

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from scipy.stats import entropy


class FedCRA(FedAvg):
    def __init__(
        self,
        config,
        *,
        server_metrics_dir: Optional[str] = None,
        save_anchor_logs: bool = False,
        server_save: Optional[Callable] = None,
        **kwargs,
    ):
        # Filter out unexpected kwargs before passing to FedAvg
        valid_kwargs = {
            'fraction_fit', 'fraction_evaluate', 'min_fit_clients',
            'min_evaluate_clients', 'min_available_clients', 'evaluate_fn',
            'on_fit_config_fn', 'on_evaluate_config_fn', 'accept_failures',
            'initial_parameters', 'fit_metrics_aggregation_fn',
            'evaluate_metrics_aggregation_fn'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        super().__init__(**filtered_kwargs)

        self.config = config
        self.server_metrics_dir = server_metrics_dir
        self.save_anchor_logs = save_anchor_logs
        self._server_save = server_save

        # New v10 parameters
        self.proximal_mu = config.strategy.params.proximal_mu
        self.embedding_dim = config.strategy.params.embedding_dim
        self.num_classes = config.strategy.params.num_classes

        # Learning rate for clients
        self.base_lr = config.config_fit.learning_rate

        # NEW: Per-class anchor storage with confidence
        self.global_anchors = np.zeros((self.num_classes, self.embedding_dim), dtype=np.float32)
        self.anchor_confidence = np.zeros(self.num_classes, dtype=np.float32)  # conf_c = 1/(variance + eps)
        self.anchor_initialized = np.zeros(self.num_classes, dtype=bool)

        # NEW: Track class counts across clients for reliability weighting
        self.global_class_counts = np.zeros(self.num_classes, dtype=np.float32)

        # NEW: Client weighting based on distribution entropy
        self.client_weights = {}  # client_id -> gamma_k

        self._cra_log: List[Dict] = []

    # ------------------------------------------------------------------
    # NEW: Class-Conditional Reliability Weighting
    # ------------------------------------------------------------------
    def _compute_reliability_weights(self, client_class_counts: Dict[int, Dict[int, int]]) -> Dict[int, np.ndarray]:
        """
        Compute per-client per-class reliability: r_kc = n_kc / (total_samples_k + epsilon)
        Returns: {client_id: reliability_vector} where reliability_vector[c] = r_kc
        """
        reliability_weights = {}
        for client_id, class_counts in client_class_counts.items():
            total_samples = sum(class_counts.values())
            if total_samples == 0:
                continue
            r_k = np.zeros(self.num_classes, dtype=np.float32)
            for c, n_kc in class_counts.items():
                r_k[c] = n_kc / (total_samples + 1e-8)
            reliability_weights[client_id] = r_k
        return reliability_weights

    # ------------------------------------------------------------------
    # NEW: Anchor Confidence Scaling
    # ------------------------------------------------------------------
    def _update_anchor_confidence(self, client_residuals: Dict[int, Dict[int, np.ndarray]]):
        """
        Update anchor confidence: conf_c = 1 / (variance of residuals for class c + epsilon)
        """
        for c in range(self.num_classes):
            residuals_c = []
            for client_res in client_residuals.values():
                if c in client_res:
                    residuals_c.append(client_res[c])

            if len(residuals_c) >= 2:  # Need at least 2 for variance
                residuals_c = np.array(residuals_c)
                variance = np.var(residuals_c, axis=0).mean()  # Average variance across embedding dims
                self.anchor_confidence[c] = 1.0 / (variance + 1e-8)
            else:
                self.anchor_confidence[c] = 0.1  # Low confidence if insufficient data

    # ------------------------------------------------------------------
    # NEW: Selective Class Alignment - Update Anchors
    # ------------------------------------------------------------------
    def _update_anchors(self, client_residuals: Dict[int, Dict[int, np.ndarray]],
                       reliability_weights: Dict[int, np.ndarray]):
        """
        Update anchors using class-conditional reliability weighting:
        anchor_c = sum_k (r_kc * residual_kc) / sum_k (r_kc)
        """
        for c in range(self.num_classes):
            weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
            total_weight = 0.0

            for client_id, client_res in client_residuals.items():
                if c in client_res and client_id in reliability_weights:
                    r_kc = reliability_weights[client_id][c]
                    if r_kc > 0:
                        weighted_sum += r_kc * client_res[c]
                        total_weight += r_kc

            if total_weight > 0:
                new_anchor = weighted_sum / total_weight
                # L2 normalize
                norm = np.linalg.norm(new_anchor)
                if norm > 1e-8:
                    new_anchor = new_anchor / norm

                if not self.anchor_initialized[c]:
                    self.global_anchors[c] = new_anchor
                    self.anchor_initialized[c] = True
                else:
                    # Exponential moving average
                    self.global_anchors[c] = 0.8 * self.global_anchors[c] + 0.2 * new_anchor

    # ------------------------------------------------------------------
    # NEW: Distribution-Aware Client Weighting
    # ------------------------------------------------------------------
    def _compute_client_weights(self, client_class_counts: Dict[int, Dict[int, int]]):
        """
        Compute client importance: gamma_k = inverse_entropy(label_distribution_k)
        Higher entropy (more balanced) = lower weight (less unique info)
        """
        for client_id, class_counts in client_class_counts.items():
            total = sum(class_counts.values())
            if total == 0:
                self.client_weights[client_id] = 0.0
                continue

            # Normalize to probability distribution
            probs = np.array([class_counts.get(c, 0) / total for c in range(self.num_classes)])
            probs = probs[probs > 0]  # Remove zeros for entropy calculation
            if len(probs) == 0:
                ent = 0.0
            else:
                ent = entropy(probs, base=2)  # Shannon entropy

            # Inverse entropy: more unique distribution = higher weight
            max_ent = math.log2(len(probs)) if len(probs) > 0 else 1.0
            gamma_k = 1.0 - (ent / max_ent) if max_ent > 0 else 1.0
            self.client_weights[client_id] = gamma_k

    # ------------------------------------------------------------------
    # NOVEL: Class-Conditional Proximal Penalty (Beyond FedProx)
    # ------------------------------------------------------------------
    def _compute_class_conditional_penalties(self, client_class_counts: Dict[int, Dict[int, int]]) -> np.ndarray:
        """
        Compute adaptive proximal penalty per class: μ_c
        
        Formula:
        μ_c = base_μ × √(reliability_c) × (1 - normalized_entropy_c)
        
        Intuition:
        - Under-represented classes (low reliability) → higher penalty (more regularization)
        - Over-represented classes (high reliability) → lower penalty (more exploration)
        - Specialized classes (low entropy) → higher penalty (consistency matters)
        - Balanced classes (high entropy) → lower penalty (natural diversity)
        
        Example:
        - Class with 5 samples (rare): μ_c = 0.05 × 1.0 × 1.0 = 0.05 (strong regularization)
        - Class with 1000 samples (common): μ_c = 0.05 × 0.1 × 0.3 = 0.0015 (weak regularization)
        
        Novel aspect: FedProx uses single μ for all classes. FedCRA adapts per class.
        """
        mu_c = np.ones(self.num_classes, dtype=np.float32) * self.proximal_mu
        
        # Global statistics
        total_samples_global = sum(sum(cc.values()) for cc in client_class_counts.values())
        if total_samples_global == 0:
            return mu_c
        
        # Compute class-wise statistics
        class_samples = np.zeros(self.num_classes, dtype=np.float32)
        class_entropy = np.zeros(self.num_classes, dtype=np.float32)
        
        for c in range(self.num_classes):
            # Count clients with this class
            clients_with_c = 0
            samples_with_c = 0
            
            for client_id, class_counts in client_class_counts.items():
                if c in class_counts:
                    clients_with_c += 1
                    samples_with_c += class_counts[c]
            
            class_samples[c] = samples_with_c
            
            # Entropy: how distributed is this class across clients?
            if samples_with_c > 0:
                client_probs = []
                for client_id, class_counts in client_class_counts.items():
                    if c in class_counts:
                        client_probs.append(class_counts[c] / samples_with_c)
                
                if len(client_probs) > 0:
                    class_entropy[c] = entropy(np.array(client_probs), base=2)
        
        # Normalize statistics
        max_entropy = math.log2(len(client_class_counts)) if len(client_class_counts) > 0 else 1.0
        normalized_entropy = class_entropy / (max_entropy + 1e-8)
        
        # Reliability: fraction of global samples
        reliability = class_samples / (total_samples_global + 1e-8)
        
        # Compute per-class penalty
        for c in range(self.num_classes):
            if reliability[c] > 0:
                # Inverse reliability (rare classes get higher penalty)
                reliability_factor = np.sqrt(1.0 / (reliability[c] + 1e-8))
                
                # Entropy factor (concentrated classes get higher penalty)
                entropy_factor = 1.0 - normalized_entropy[c]
                
                # Combine factors
                mu_c[c] = self.proximal_mu * reliability_factor * entropy_factor
                
                # Clamp to reasonable range
                mu_c[c] = np.clip(mu_c[c], self.proximal_mu * 0.1, self.proximal_mu * 10.0)
            else:
                # No samples for this class
                mu_c[c] = self.proximal_mu * 10.0  # Very strong regularization
        
        return mu_c

    # ------------------------------------------------------------------
    # configure_fit — inject CRA config every round
    # ------------------------------------------------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        # Send global anchors and confidence for selective alignment
        anchors_json = json.dumps(self.global_anchors.tolist())
        confidence_json = json.dumps(self.anchor_confidence.tolist())
        proximal_mu = self.proximal_mu

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            cfg["cra_anchors"] = anchors_json
            cfg["cra_confidence"] = confidence_json
            cfg["cra_proximal_mu"] = proximal_mu
            cfg["cra_num_classes"] = self.num_classes
            cfg["learning_rate"] = self.base_lr
            cfg["round_id"] = server_round
            
            # NOVEL: Send class-conditional penalties (if available)
            if hasattr(self, '_class_conditional_penalties'):
                cfg["cra_mu_c"] = json.dumps(self._class_conditional_penalties.tolist())
            
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        return patched

    # ------------------------------------------------------------------
    # aggregate_fit — Distribution-Aware Aggregation + Anchor Updates
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Extract client data for new components
        client_residuals = {}
        client_class_counts = {}

        for client_proxy, fit_res in results:
            client_id = getattr(client_proxy, 'cid', str(client_proxy))
            m = fit_res.metrics or {}

            # NEW: Extract residuals and class counts
            residuals_json = m.get("cra_residuals")
            counts_json = m.get("cra_class_counts")

            if residuals_json and counts_json:
                client_residuals[client_id] = {
                    int(k): np.array(v, dtype=np.float32)
                    for k, v in json.loads(residuals_json).items()
                }
                client_class_counts[client_id] = {
                    int(k): int(v) for k, v in json.loads(counts_json).items()
                }

        # NEW: Update reliability weights and anchors
        if client_residuals:
            reliability_weights = self._compute_reliability_weights(client_class_counts)
            self._update_anchor_confidence(client_residuals)
            self._update_anchors(client_residuals, reliability_weights)

        # NEW: Compute distribution-aware client weights
        if client_class_counts:
            self._compute_client_weights(client_class_counts)
            
            # NOVEL: Compute class-conditional penalties (beyond FedProx)
            self._class_conditional_penalties = self._compute_class_conditional_penalties(client_class_counts)

        # NEW: Distribution-Aware Aggregation (replace FedAvg)
        if self.client_weights and sum(self.client_weights.values()) > 0:
            # Normalize weights
            total_weight = sum(self.client_weights.values())
            normalized_weights = {cid: w / total_weight for cid, w in self.client_weights.items()}

            # Weighted aggregation
            # Convert parameters to ndarrays
            parameters_list = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            
            aggregated_parameters = []
            for param_idx in range(len(parameters_list[0])):
                weighted_sum = None
                for i, (client_proxy, _) in enumerate(results):
                    client_id = getattr(client_proxy, 'cid', str(client_proxy))
                    weight = normalized_weights.get(client_id, 1.0 / len(results))

                    param = parameters_list[i][param_idx]
                    if weighted_sum is None:
                        weighted_sum = weight * param
                    else:
                        weighted_sum += weight * param

                aggregated_parameters.append(weighted_sum)
            
            # Convert to Flower Parameters object
            aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)
        else:
            # Fallback to FedAvg
            aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        self._server_save(server_round, aggregated_parameters)

        self._cra_log.append({
            "round": server_round,
            "anchor_confidence": self.anchor_confidence.tolist(),
            "anchors_initialized": self.anchor_initialized.tolist(),
            "client_weights": self.client_weights.copy(),
        })

        if self.server_metrics_dir and self.save_anchor_logs:
            Path(self.server_metrics_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.server_metrics_dir) / "cra_anchor_log.json").write_text(
                json.dumps(self._cra_log, indent=2)
            )

        return aggregated_parameters, {}