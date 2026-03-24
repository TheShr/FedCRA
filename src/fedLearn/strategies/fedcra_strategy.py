"""
FedCRA: Federated Class-Residual Anchoring Strategy  v4
---------------------------------------------------------
Clean design — no pretrain phase, no warmup gimmicks.

Key principles:
  1. CRA loss is active from round 1, but alpha starts SMALL (0.1) and
     grows to peak (0.4) over the first 10 rounds, then decays back to 0.05.
     This triangular schedule: ramp-up -> peak -> decay avoids both the
     "zero-anchor pull" problem (round 1) and the "plateau" problem (later).

  2. LR is constant at base_lr — no cosine decay.
     Cosine decay was causing the plateau because it reduced the effective
     learning signal faster than the CRA loss could compensate. Let the
     Adam optimiser handle adaptation instead.

  3. gradient_clip=1.0 is enforced on the client side to stop the CRA
     loss from causing gradient explosions when anchors first become active.

  4. Anchor EMA: anchors are updated with exponential moving average
     (momentum=0.9) rather than hard replacement. This makes anchors
     stable even when only a few clients see a minority class in one round.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg


class FedCRA(FedAvg):
    def __init__(
        self,
        *,
        alpha_cra_peak: float = 0.4,    # peak CRA loss weight
        alpha_cra_min: float = 0.05,    # floor CRA loss weight
        alpha_ramp_rounds: int = 10,    # rounds to reach peak from 0
        lambda_severity: float = 4.0,
        embedding_dim: int = 128,
        num_classes: int = 9,
        anchor_momentum: float = 0.9,   # EMA momentum for anchor updates
        grad_clip: float = 1.0,         # gradient clip norm sent to clients
        base_lr: float = 0.001,
        total_rounds: int = 50,
        server_metrics_dir: Optional[str] = None,
        server_save: Optional[Callable] = None,
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)
        self.alpha_cra_peak = alpha_cra_peak
        self.alpha_cra_min = alpha_cra_min
        self.alpha_ramp_rounds = alpha_ramp_rounds
        self.lambda_severity = lambda_severity
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.anchor_momentum = anchor_momentum
        self.grad_clip = grad_clip
        self.base_lr = base_lr
        self.total_rounds = total_rounds
        self.server_metrics_dir = server_metrics_dir
        self._server_save = server_save

        self.global_anchors = np.zeros((num_classes, embedding_dim), dtype=np.float32)
        self.rho = np.ones(num_classes, dtype=np.float32)
        self._anchors_initialised = np.zeros(num_classes, dtype=bool)
        self._cra_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Triangular alpha schedule:
    #   rounds 1..alpha_ramp_rounds   -> linear ramp 0 -> peak
    #   rounds alpha_ramp_rounds..end -> cosine decay peak -> min
    # This avoids pulling toward zero anchors in round 1 (alpha=0),
    # hits maximum minority protection at round alpha_ramp_rounds,
    # then relaxes so accuracy can keep climbing.
    # ------------------------------------------------------------------
    def _compute_alpha(self, server_round: int) -> float:
        if server_round <= self.alpha_ramp_rounds:
            # linear ramp: 0 at round 1, peak at alpha_ramp_rounds
            progress = (server_round - 1) / max(1, self.alpha_ramp_rounds - 1)
            return self.alpha_cra_min + (self.alpha_cra_peak - self.alpha_cra_min) * progress
        else:
            # cosine decay from peak to min over remaining rounds
            post = server_round - self.alpha_ramp_rounds
            total_post = max(1, self.total_rounds - self.alpha_ramp_rounds)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * post / total_post))
            return self.alpha_cra_min + (self.alpha_cra_peak - self.alpha_cra_min) * cosine_decay

    # ------------------------------------------------------------------
    # EMA anchor update — stable even with sparse minority observations
    # ------------------------------------------------------------------
    def _update_anchors(self, client_centroids, client_counts):
        N = len(client_centroids)
        new_rho = np.ones(self.num_classes, dtype=np.float32)

        for k in range(self.num_classes):
            weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
            total_weight = 0.0
            n_k_clients = 0

            for centroids, counts in zip(client_centroids, client_counts):
                if k in centroids and k in counts and counts[k] > 0:
                    w = float(counts[k])
                    weighted_sum += w * np.array(centroids[k], dtype=np.float32)
                    total_weight += w
                    n_k_clients += 1

            if total_weight > 0:
                new_centroid = weighted_sum / total_weight
                if not self._anchors_initialised[k]:
                    # First time seeing this class: hard initialise
                    self.global_anchors[k] = new_centroid
                    self._anchors_initialised[k] = True
                else:
                    # EMA update — smooths out round-to-round noise
                    m = self.anchor_momentum
                    self.global_anchors[k] = m * self.global_anchors[k] + (1 - m) * new_centroid

            new_rho[k] = math.exp(-self.lambda_severity * n_k_clients / max(1, N))

        self.rho = new_rho

    # ------------------------------------------------------------------
    # configure_fit — inject CRA config every round
    # ------------------------------------------------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        alpha = self._compute_alpha(server_round)
        anchors_json = json.dumps(self.global_anchors.tolist())
        rho_json = json.dumps(self.rho.tolist())

        # Class-balanced CE weights derived from minority severity.
        # Rare classes (high rho) get upweighted CE loss, improving F1/Precision/Recall
        # by preventing the CE loss from collapsing toward the majority class.
        # Normalised so mean weight = 1.0 (preserves overall loss magnitude).
        raw_w = 1.0 + self.rho                          # range [1, 2] — rare=2, common=1
        class_weights = (raw_w / raw_w.mean()).tolist()
        class_weights_json = json.dumps(class_weights)

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            cfg["cra_anchors"] = anchors_json
            cfg["cra_rho"] = rho_json
            cfg["cra_alpha"] = alpha
            cfg["cra_num_classes"] = self.num_classes
            cfg["cra_grad_clip"] = self.grad_clip
            cfg["cra_class_weights"] = class_weights_json
            cfg["learning_rate"] = self.base_lr
            cfg["round_id"] = server_round
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        return patched

    # ------------------------------------------------------------------
    # aggregate_fit
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        client_centroids, client_counts = [], []
        for _, fit_res in results:
            m = fit_res.metrics or {}
            c_json = m.get("cra_centroids")
            n_json = m.get("cra_counts")
            if c_json and n_json:
                client_centroids.append({int(k): v for k, v in json.loads(c_json).items()})
                client_counts.append({int(k): int(v) for k, v in json.loads(n_json).items()})

        if client_centroids:
            self._update_anchors(client_centroids, client_counts)

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures)

        if aggregated_parameters is not None and self._server_save is not None:
            self._server_save(server_round, aggregated_parameters)

        self._cra_log.append({
            "round": server_round,
            "alpha_cra": self._compute_alpha(server_round),
            "rho": self.rho.tolist(),
            "anchors_initialised": self._anchors_initialised.tolist(),
        })
        if self.server_metrics_dir:
            Path(self.server_metrics_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.server_metrics_dir) / "cra_anchor_log.json").write_text(
                json.dumps(self._cra_log, indent=2))

        return aggregated_parameters, aggregated_metrics
