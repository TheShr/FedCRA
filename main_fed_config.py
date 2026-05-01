import os
import random
import hydra
import pickle
import time
import json
import warnings
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters

from src.fedLearn.server.server_side import get_on_fit_config
from src.fedLearn.client import generate_client_fn
from src.fedLearn.fed_data import federated_data_dirichlet
from log_config import base_logger

warnings.filterwarnings("ignore",
    message=re.escape("This DataLoader will create") + ".*",
    category=UserWarning, module="torch.utils.data.dataloader")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
torch.set_default_tensor_type(torch.FloatTensor)
logger = base_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seed set to {seed}")


def model_to_parameters(model):
    return ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()])


def evaluate_server_model(model, test_loader, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += crit(logits, yb).item() * yb.size(0)
            pred = logits.argmax(1)
            total += yb.size(0)
            all_y.extend(yb.cpu().numpy())
            all_p.extend(pred.cpu().numpy())

    from src.fedLearn.centralized import compute_macro_fpr, compute_per_class_f1
    acc = accuracy_score(all_y, all_p)
    precision_macro = precision_score(all_y, all_p, average="macro", zero_division=0)
    recall_macro = recall_score(all_y, all_p, average="macro", zero_division=0)
    f1_macro = f1_score(all_y, all_p, average="macro", zero_division=0)
    f1_weighted = f1_score(all_y, all_p, average="weighted", zero_division=0)
    labels = np.arange(len(class_names)) if class_names is not None else None
    precision_per_class = precision_score(all_y, all_p, average=None, labels=labels, zero_division=0)
    recall_per_class = recall_score(all_y, all_p, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(all_y, all_p, average=None, labels=labels, zero_division=0)

    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1_score": f1_macro,
        "f1_weighted": f1_weighted,
        "macro_fpr": compute_macro_fpr(all_y, all_p),
        "per_class_f1": compute_per_class_f1(all_y, all_p, num_classes=len(class_names) if class_names is not None else None),
    }

    class_names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(len(f1_per_class))]
    for idx, cls in enumerate(class_names):
        label_key = str(cls).replace(' ', '_').lower()
        metrics[f"{label_key}_precision"] = float(precision_per_class[idx])
        metrics[f"{label_key}_recall"] = float(recall_per_class[idx])
        metrics[f"{label_key}_f1"] = float(f1_per_class[idx])

    return metrics


def save_server_metrics_json(metrics, server_round, server_metrics_dir, dt):
    d = Path(server_metrics_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "server_metrics.json"
    data = json.loads(f.read_text()) if f.exists() else []
    row = {"round": server_round, "communication_time": dt}
    for k, v in metrics.items():
        row[k] = v
    data.append(row)
    f.write_text(json.dumps(data, indent=4))


def get_evaluate_server_fn(model, test_loader, server_metrics_dir, class_names=None):
    def evaluate_fn(server_round, parameters, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in
             zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict, strict=True)
        t0 = time.time()
        metrics = evaluate_server_model(model, test_loader, class_names=class_names)
        save_server_metrics_json(metrics, server_round, server_metrics_dir,
                                 time.time() - t0)
        return metrics["loss"], {"accuracy": metrics["accuracy"]}
    return evaluate_fn


def _make_save_fn(model, server_model_dir):
    def _save(server_round, aggregated_parameters):
        # Check available disk space before saving
        try:
            stat = os.statvfs(server_model_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if available_gb < 1.0:  # Less than 1GB available
                logger.warning(f"Low disk space ({available_gb:.3f}GB available). Skipping model save at round {server_round}.")
                return
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}. Proceeding with save.")

        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in
             zip(model.state_dict().keys(), ndarrays)})
        model.load_state_dict(state_dict, strict=True)
        Path(server_model_dir).mkdir(parents=True, exist_ok=True)
        model.to("cpu")
        torch.save(model.state_dict(),
                   Path(server_model_dir) / f"sm_{server_round}.pth")
        logger.info(f"Server model saved: sm_{server_round}.pth")
    return _save


# ── Cube-root weighting mixin ──────────────────────────────────────────────
# Applied to ALL strategies (FedAvg, FedProx, etc.) that go through the
# generic build path below.  FedCRA handles its own cube-root internally.
#
# Why cube-root?
#   sqrt  dampens a 10k:8 ratio from 1250x → 35x  (over-corrects at alpha=0.1)
#   cbrt  dampens a 10k:8 ratio from 1250x → 11x  (gentler, more stable)
# ──────────────────────────────────────────────────────────────────────────
class _CuberootMixin:
    """Replace num_examples with cube-root before FedAvg weighted aggregation."""

    def aggregate_fit(self, server_round, results, failures):
        cuberoot_results = []
        for client_proxy, fit_res in results:
            fair_weight = max(1, round(fit_res.num_examples ** (1.0 / 3.0)))
            cuberoot_results.append((
                client_proxy,
                fl.common.FitRes(
                    status=fit_res.status,
                    parameters=fit_res.parameters,
                    num_examples=fair_weight,
                    metrics=fit_res.metrics,
                )
            ))
        return super().aggregate_fit(server_round, cuberoot_results, failures)


def build_strategy(cfg: DictConfig, model, evaluate_fn, server_model_dir: str, server_metrics_dir: str):
    strategy_name = cfg.strategy.name
    extra_params = (dict(cfg.strategy.params)
                    if "params" in cfg.strategy and cfg.strategy.params
                    else {})
    save_fn = _make_save_fn(model, server_model_dir)

    common = dict(
        fraction_fit=cfg.fed_config.fraction_fit,
        min_fit_clients=cfg.fed_config.num_clients_per_round_fit,
        fraction_evaluate=cfg.fed_config.fraction_eval,
        min_evaluate_clients=cfg.fed_config.num_clients_per_round_eval,
        min_available_clients=cfg.fed_config.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=evaluate_fn,
        initial_parameters=model_to_parameters(model),
    )

    # ── FedCRA: cube-root is handled inside FedCRA.aggregate_fit ──────────
    if strategy_name == "FedCRA":
        from src.fedLearn.strategies.fedcra_strategy import FedCRA
        return FedCRA(
            cfg,
            server_metrics_dir=str(server_metrics_dir),
            save_anchor_logs=True,
            server_save=save_fn,
            **common,
        )

    # ── FedProx: Custom implementation ─────────────────────────────────
    if strategy_name == "FedProx":
        from src.fedLearn.strategies.fedprox_strategy import FedProx
        return FedProx(
            cfg,
            server_metrics_dir=str(server_metrics_dir),
            server_save=save_fn,
            **common,
        )

    # ── Custom strategy implementations for requested comparison strategies ──
    custom_strategy_map = {
        "FedScaffold": "FedSCaffold",
        "FedSCaffold": "FedSCaffold",
        "FedLC": "FedLC",
        "FedFocal": "FedFocal",
        "FedBB": "FedBB",
        "FedLTA": "FedLTA",
        "FLAME": "FLAME",
    }

    if strategy_name in custom_strategy_map:
        from src.fedLearn.strategies.custom_strategies import (
            FedBB, FedFocal, FedLC, FedLTA, FedSCaffold, FLAME,
        )
        canonical_name = custom_strategy_map[strategy_name]
        strategy_cls = {
            "FedSCaffold": FedSCaffold,
            "FedLC": FedLC,
            "FedFocal": FedFocal,
            "FedBB": FedBB,
            "FedLTA": FedLTA,
            "FLAME": FLAME,
        }[canonical_name]
        return strategy_cls(
            cfg,
            server_metrics_dir=str(server_metrics_dir),
            server_save=save_fn,
            **common,
            **extra_params,
        )

    # ── All other strategies: inject _CuberootMixin + _SaveMixin ──────────
    class _SaveMixin:
        def __init__(self, *a, _save_fn=None, **kw):
            self.__save_fn = _save_fn
            super().__init__(*a, **kw)

        def aggregate_fit(self, server_round, results, failures):
            agg_params, agg_metrics = super().aggregate_fit(
                server_round, results, failures)
            if agg_params is not None and self.__save_fn:
                try:
                    self.__save_fn(server_round, agg_params)
                except Exception as e:
                    logger.warning(f"Failed to save model checkpoint at round {server_round}: {e}")
            return agg_params, agg_metrics

    strategy_cls = getattr(fl.server.strategy, strategy_name, None)
    if strategy_cls is None:
        avail = [n for n in dir(fl.server.strategy) if n.startswith("Fed")]
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {avail}")

    Mixed = type("_Strategy", (_SaveMixin, _CuberootMixin, strategy_cls), {})
    return Mixed(**common, _save_fn=save_fn)

    # ── All other strategies: inject _CuberootMixin + _SaveMixin ──────────
    class _SaveMixin:
        def __init__(self, *a, _save_fn=None, **kw):
            self.__save_fn = _save_fn
            super().__init__(*a, **kw)

        def aggregate_fit(self, server_round, results, failures):
            agg_params, agg_metrics = super().aggregate_fit(
                server_round, results, failures)
            if agg_params is not None and self.__save_fn:
                self.__save_fn(server_round, agg_params)
            return agg_params, agg_metrics

    strategy_cls = getattr(fl.server.strategy, strategy_name, None)
    if strategy_cls is None:
        avail = [n for n in dir(fl.server.strategy) if n.startswith("Fed")]
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {avail}")

    Mixed = type("_Strategy", (_SaveMixin, _CuberootMixin, strategy_cls), {})
    return Mixed(**common, _save_fn=save_fn)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    OmegaConf.set_struct(cfg, False)
    cfg.config_fit.cra_grad_clip = cfg.get("cra_grad_clip", cfg.config_fit.get("cra_grad_clip", 1.0))
    print(OmegaConf.to_yaml(cfg))

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_folder  = to_absolute_path(cfg.data_config.folder_name)
    results_base = to_absolute_path(cfg.fed_config.model_results_path)

    logger.info(f"Data folder   = {data_folder}")
    logger.info(f"Results base  = {results_base}")

    (client_train_loaders, client_val_loaders,
     serv_train_loader, ser_test_loader,
     detected_num_classes, class_names) = federated_data_dirichlet(
        data_folder=data_folder,
        data_file=cfg.data_config.file_name,
        label_name=cfg.data_config.label_name,
        n_features=cfg.data_config.n_features,
        num_clients=cfg.fed_config.num_clients,
        train_batch_size=cfg.fed_config.train_batch_size,
        alpha=cfg.dirichlet.alpha,
        total_samples=cfg.data_config.total_samples,
        class_ratios=cfg.imbalance.class_ratios,
        seed=seed,
    )

    cfg.dataset.num_classes = detected_num_classes
    if cfg.strategy.name == "FedCRA":
        if "params" not in cfg.strategy or cfg.strategy.params is None:
            cfg.strategy.params = {}
        cfg.strategy.params.num_classes = detected_num_classes

    model = instantiate(cfg.model).to(device)
    model_name = model.__class__.__name__

    strategy_name    = cfg.strategy.name
    model_root       = Path(results_base) / strategy_name / model_name
    server_model_dir = model_root / "server"
    server_metrics_dir = model_root / "metrics"
    server_model_dir.mkdir(parents=True, exist_ok=True)
    server_metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Strategy      = {strategy_name}")
    logger.info(f"Model root    = {model_root}")

    client_names = [f"c{i+1}" for i in range(cfg.fed_config.num_clients)]
    client_fn = generate_client_fn(
        model=model,
        train_loaders=client_train_loaders,
        test_loaders=client_val_loaders,
        client_names=client_names,
        results_path=str(model_root),
    )

    strategy = build_strategy(
        cfg=cfg,
        model=model,
        evaluate_fn=get_evaluate_server_fn(
            model=model,
            test_loader=ser_test_loader,
            server_metrics_dir=str(server_metrics_dir),
            class_names=class_names,
        ),
        server_model_dir=str(server_model_dir),
        server_metrics_dir=str(server_metrics_dir),
    )

    per_client_cpus = float(cfg.fed_config.num_cpus) / max(1, cfg.fed_config.num_clients)
    # Optimize GPU allocation for 12GB GPU: give each client more GPU for faster training
    # With 10 clients, each gets ~1.2GB GPU memory (more than enough for DNN)
    per_client_gpus = 0.15  # 1.8GB per client total, allows ~5-6 parallel clients
    
    logger.info(f"Starting federated learning simulation...")
    logger.info(f"  Clients: {cfg.fed_config.num_clients}")
    logger.info(f"  Rounds: {cfg.fed_config.num_rounds}")
    logger.info(f"  Strategy: {cfg.strategy.name}")
    logger.info(f"  CPUs per client: {per_client_cpus}")
    logger.info(f"  GPUs per client: {per_client_gpus}")
    
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.fed_config.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.fed_config.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": per_client_cpus,
                "num_gpus": per_client_gpus,
            },
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": cfg.fed_config.num_cpus,
            },
        )
    except Exception as e:
        logger.error(f"Simulation failed with error: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    hist_file = model_root / f"history_{cfg.dataset.dataset_name}.pkl"
    with open(hist_file, "wb") as h:
        pickle.dump({"history": history}, h, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"History saved: {hist_file}")


if __name__ == "__main__":
    main()