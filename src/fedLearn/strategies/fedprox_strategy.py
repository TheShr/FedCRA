"""
FedProx: Federated Optimization in Heterogeneous Networks
Li et al., 2020 - Standard implementation for baseline comparison.

Proximal term: μ/2 ||w - w_g||²
- Stabilizes learning in non-IID settings
- Single penalty μ for all parameters and classes
- Baseline for comparing FedCRA's class-conditional improvements
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg


class FedProx(FedAvg):
    def __init__(
        self,
        config,
        *,
        server_metrics_dir: Optional[str] = None,
        server_save: Optional[Callable] = None,
        **kwargs,
    ):
        """
        FedProx Strategy
        
        Args:
            config: Configuration object with strategy.params.proximal_mu
            server_metrics_dir: Directory to save server metrics
            server_save: Callback for saving results
            **kwargs: Additional arguments for FedAvg
        """
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
        self._server_save = server_save

        # FedProx parameters
        self.proximal_mu = config.strategy.params.proximal_mu
        self.base_lr = config.config_fit.learning_rate
        
        self._round_counter = 0

    # ------------------------------------------------------------------
    # configure_fit — Send proximal penalty and other config
    # ------------------------------------------------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            cfg["proximal_mu"] = self.proximal_mu  # Global penalty
            cfg["learning_rate"] = self.base_lr
            cfg["round_id"] = server_round
            
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        return patched

    # ------------------------------------------------------------------
    # aggregate_fit — Standard FedAvg aggregation (no class-specific logic)
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round, results, failures):
        """
        Standard FedAvg aggregation.
        Unlike FedCRA, FedProx does not compute class-conditional penalties or anchors.
        The proximal term is applied at the client side during training.
        """
        if not results:
            return None, {}

        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        self._round_counter += 1

        # Log metrics if available
        if self.server_metrics_dir and aggregated_metrics:
            metrics_dir = Path(self.server_metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"round_{server_round}_metrics.json"
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(aggregated_metrics, f, indent=2)
            except Exception as e:
                print(f"[FedProx] Failed to save metrics: {e}")

        return aggregated_parameters, aggregated_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate — Standard evaluation (no special handling)
    # ------------------------------------------------------------------
    def aggregate_evaluate(self, server_round, results, failures):
        """
        Standard FedAvg evaluation aggregation.
        """
        if not results:
            return None, {}

        return super().aggregate_evaluate(server_round, results, failures)
