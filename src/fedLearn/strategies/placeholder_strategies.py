"""Placeholder strategy implementations for experiment comparison.

These strategy classes are currently thin wrappers around Flower's FedAvg
server strategy. They allow the framework to run experiments and compare
FedCRA against strategy names such as FedSCaffold, FedLC, FedFocal, FedBB,
FedLTA, and FLAME while preserving the existing aggregation path.

The exact algorithmic implementations can be added later by expanding these
classes with strategy-specific client or server logic.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import flwr as fl
from flwr.server.strategy import FedAvg


class FedPlaceholder(FedAvg):
    """Generic placeholder strategy based on FedAvg."""

    def __init__(
        self,
        *args,
        strategy_name: Optional[str] = None,
        server_metrics_dir: Optional[str] = None,
        server_save: Optional[Callable] = None,
        **kwargs,
    ):
        valid_kwargs = {
            'fraction_fit', 'fraction_evaluate', 'min_fit_clients',
            'min_evaluate_clients', 'min_available_clients', 'evaluate_fn',
            'on_fit_config_fn', 'on_evaluate_config_fn', 'accept_failures',
            'initial_parameters', 'fit_metrics_aggregation_fn',
            'evaluate_metrics_aggregation_fn',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        super().__init__(**filtered_kwargs)

        self.strategy_name = strategy_name or self.__class__.__name__
        self.server_metrics_dir = server_metrics_dir
        self._server_save = server_save

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if self.server_metrics_dir and aggregated_metrics:
            metrics_dir = Path(self.server_metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(metrics_dir / f"{self.strategy_name.lower()}_round_{server_round}_metrics.json", 'w') as fp:
                    json.dump(aggregated_metrics, fp, indent=2)
            except Exception:
                pass

        return aggregated_parameters, aggregated_metrics


class FedSCaffold(FedPlaceholder):
    """Placeholder for SCAFFOLD-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FedSCaffold", *args, **kwargs)


class FedLC(FedPlaceholder):
    """Placeholder for FedLC-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FedLC", *args, **kwargs)


class FedFocal(FedPlaceholder):
    """Placeholder for FedFocal-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FedFocal", *args, **kwargs)


class FedBB(FedPlaceholder):
    """Placeholder for FedBB-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FedBB", *args, **kwargs)


class FedLTA(FedPlaceholder):
    """Placeholder for FedLTA-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FedLTA", *args, **kwargs)


class FLAME(FedPlaceholder):
    """Placeholder for FLAME-like strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(strategy_name="FLAME", *args, **kwargs)
