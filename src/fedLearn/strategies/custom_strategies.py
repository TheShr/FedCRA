"""Custom federated strategy implementations for paper-based comparison.

These strategies implement distinct algorithmic behavior so the methods
are not mere FedAvg placeholders.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


def _ndarray_list_to_json(ndarrays: List[np.ndarray]) -> str:
    return json.dumps([arr.tolist() for arr in ndarrays])


def _json_to_ndarray_list(json_str: str) -> List[np.ndarray]:
    return [np.array(arr, dtype=np.float32) for arr in json.loads(json_str)]


def _weighted_average(params_list: List[List[np.ndarray]], weights: List[float]) -> List[np.ndarray]:
    total_weight = sum(weights)
    if total_weight <= 0:
        return params_list[0]
    averaged = []
    for layer_params in zip(*params_list):
        layer_sum = sum(p * w for p, w in zip(layer_params, weights))
        averaged.append(layer_sum / total_weight)
    return averaged


class BaseCustomStrategy(FedAvg):
    def __init__(
        self,
        strategy_name: str,
        server_metrics_dir: Optional[str] = None,
        server_save: Optional[Callable] = None,
        **kwargs,
    ):
        valid_kwargs = {
            'fraction_fit', 'fraction_evaluate', 'min_fit_clients',
            'min_evaluate_clients', 'min_available_clients', 'evaluate_fn',
            'on_fit_config_fn', 'on_evaluate_config_fn', 'accept_failures',
            'initial_parameters', 'fit_metrics_aggregation_fn',
            'evaluate_metrics_aggregation_fn'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        super().__init__(**filtered_kwargs)

        self.strategy_name = strategy_name
        self.name = strategy_name
        self.server_metrics_dir = server_metrics_dir
        self._server_save = server_save

    def _save_metrics(self, server_round: int, metrics: Dict):
        if self.server_metrics_dir and metrics:
            metrics_dir = Path(self.server_metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            with open(metrics_dir / f"{self.strategy_name.lower()}_round_{server_round}_metrics.json", 'w') as fp:
                json.dump(metrics, fp, indent=2)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None and self._server_save:
            self._server_save(server_round, aggregated_parameters)
        self._save_metrics(server_round, aggregated_metrics)
        return aggregated_parameters, aggregated_metrics


class FedSCaffold(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FedSCaffold",
        server_metrics_dir: Optional[str] = None,
        server_save: Optional[Callable] = None,
        control_variate_lr: float = 1.0,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, server_metrics_dir=server_metrics_dir,
                         server_save=server_save, **kwargs)
        self.control_variate_lr = control_variate_lr
        self.global_control_variate = None
        self.client_control_variates = {}
        self.previous_global_params = None

    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        if self.global_control_variate is None:
            self.global_control_variate = [np.zeros_like(ndarray) for ndarray in parameters_to_ndarrays(parameters)]

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            client_id = str(client_proxy.cid)
            local_c = self.client_control_variates.get(client_id)
            if local_c is None:
                local_c = [np.zeros_like(arr) for arr in self.global_control_variate]
            cfg["use_scaffold"] = True
            cfg["client_id"] = client_id
            cfg["scaffold_c_global"] = _ndarray_list_to_json(self.global_control_variate)
            cfg["scaffold_c_local"] = _ndarray_list_to_json(local_c)
            cfg["scaffold_control_variate_lr"] = self.control_variate_lr
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        self.previous_global_params = parameters_to_ndarrays(parameters)
        return patched

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        client_param_lists = []
        client_weights = []
        delta_controls = []

        for _, fit_res in results:
            client_param_lists.append(parameters_to_ndarrays(fit_res.parameters))
            client_weights.append(fit_res.num_examples)
            scaffold_json = fit_res.metrics.get("scaffold_c_local") if fit_res.metrics else None
            if scaffold_json:
                client_id = str(fit_res.metrics.get("client_id", ""))
                local_c = _json_to_ndarray_list(scaffold_json)
                prev_local = self.client_control_variates.get(client_id, [np.zeros_like(arr) for arr in local_c])
                delta_controls.append((client_id, [new - old for new, old in zip(local_c, prev_local)]))
                self.client_control_variates[client_id] = local_c

        aggregated = _weighted_average(client_param_lists, client_weights)
        aggregated_parameters = ndarrays_to_parameters(aggregated)

        if delta_controls:
            total_clients = len(delta_controls)
            if total_clients > 0:
                delta_mean = []
                num_layers = len(delta_controls[0][1])
                for layer_idx in range(num_layers):
                    layer_deltas = [delta_list[layer_idx] for _, delta_list in delta_controls]
                    delta_mean.append(np.mean(layer_deltas, axis=0))
                self.global_control_variate = [g + d for g, d in zip(self.global_control_variate, delta_mean)]

        aggregated_parameters, aggregated_metrics = aggregated_parameters, {}
        if self._server_save:
            self._server_save(server_round, aggregated_parameters)
        if self.server_metrics_dir and aggregated_metrics:
            self._save_metrics(server_round, aggregated_metrics)
        return aggregated_parameters, aggregated_metrics


class FedFocal(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FedFocal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            cfg["use_focal_loss"] = True
            cfg["focal_gamma"] = float(self.focal_gamma)
            cfg["focal_alpha"] = float(self.focal_alpha)
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        return patched


class FedLC(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FedLC",
        fedlc_mu: float = 0.001,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, **kwargs)
        self.fedlc_mu = fedlc_mu

    def configure_fit(self, server_round, parameters, client_manager):
        configs = super().configure_fit(server_round, parameters, client_manager)
        if not configs:
            return configs

        patched = []
        for client_proxy, fit_ins in configs:
            cfg = dict(fit_ins.config)
            cfg["use_fedlc"] = True
            cfg["fedlc_mu"] = float(self.fedlc_mu)
            patched.append(
                (client_proxy, fl.common.FitIns(parameters=fit_ins.parameters, config=cfg))
            )
        return patched


class FedBB(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FedBB",
        bb_beta: float = 0.5,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, **kwargs)
        self.bb_beta = bb_beta

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        param_lists = []
        weights = []
        metrics_agg = {}
        for _, fit_res in results:
            param_lists.append(parameters_to_ndarrays(fit_res.parameters))
            loss = float(fit_res.metrics.get("loss", 1.0) if fit_res.metrics else 1.0)
            weight = np.exp(-self.bb_beta * loss) * fit_res.num_examples
            weights.append(weight)

        aggregated = _weighted_average(param_lists, weights)
        aggregated_parameters = ndarrays_to_parameters(aggregated)
        aggregated_parameters, aggregated_metrics = aggregated_parameters, {}

        if self._server_save:
            self._server_save(server_round, aggregated_parameters)
        self._save_metrics(server_round, aggregated_metrics)
        return aggregated_parameters, aggregated_metrics


class FedLTA(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FedLTA",
        lta_beta: float = 0.5,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, **kwargs)
        self.lta_beta = lta_beta
        self.previous_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        param_lists = []
        weights = []
        for _, fit_res in results:
            param_lists.append(parameters_to_ndarrays(fit_res.parameters))
            weights.append(fit_res.num_examples)

        if self.previous_parameters is None:
            aggregated = _weighted_average(param_lists, weights)
        else:
            aggregated = []
            layer_deltas = [np.zeros_like(p) for p in self.previous_parameters]
            avg_params = _weighted_average(param_lists, weights)
            for idx, (avg_param, prev_param) in enumerate(zip(avg_params, self.previous_parameters)):
                delta = avg_param - prev_param
                layer_norm = np.linalg.norm(delta)
                adaptive_step = 1.0 / (1.0 + self.lta_beta * layer_norm)
                aggregated.append(prev_param + adaptive_step * delta)
        aggregated_parameters = ndarrays_to_parameters(aggregated)
        self.previous_parameters = aggregated

        if self._server_save:
            self._server_save(server_round, aggregated_parameters)
        self._save_metrics(server_round, {})
        return aggregated_parameters, {}


class FLAME(BaseCustomStrategy):
    def __init__(
        self,
        strategy_name: str = "FLAME",
        flame_alpha: float = 0.7,
        **kwargs,
    ):
        super().__init__(strategy_name=strategy_name, **kwargs)
        self.flame_alpha = flame_alpha
        self.ensemble_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        param_lists = []
        weights = []
        for _, fit_res in results:
            param_lists.append(parameters_to_ndarrays(fit_res.parameters))
            weights.append(fit_res.num_examples)

        aggregated = _weighted_average(param_lists, weights)
        if self.ensemble_parameters is None:
            self.ensemble_parameters = aggregated
        else:
            self.ensemble_parameters = [
                self.flame_alpha * e + (1.0 - self.flame_alpha) * a
                for e, a in zip(self.ensemble_parameters, aggregated)
            ]
        aggregated_parameters = ndarrays_to_parameters(self.ensemble_parameters)

        if self._server_save:
            self._server_save(server_round, aggregated_parameters)
        self._save_metrics(server_round, {})
        return aggregated_parameters, {}
