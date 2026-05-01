#!/usr/bin/env python3
"""
Imbalanced Split Experiments for FedCRA

Runs federated learning experiments with 2 clients using imbalanced data splits.
Measures training time, testing time, communication cost, model size, and extra parameters.
"""

import csv
import os
import time
import json
import pickle
import argparse
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import flwr as fl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from data_imbalance import create_imbalance, split_non_iid_clients, train_test_split_stratified
from src.fedLearn.client import generate_client_fn
from log_config import base_logger
from main_fed_config import (
    set_global_seed, build_strategy, get_evaluate_server_fn,
    evaluate_server_model, model_to_parameters
)

logger = base_logger(__name__)

class ComprehensiveLogger:
    """Logs comprehensive experiment information in structured format."""
    
    def __init__(self, log_file_path):
        self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        
    def write_header(self, cfg, ratios, strategies, num_rounds):
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Imbalanced Split Experiments\n")
            f.write("="*80 + "\n")
            f.write(f"Start Time: {self.start_time.strftime('%a %b %d %H:%M:%S %Z %Y')}\n")
            f.write(f"Project Path: {Path(__file__).parent}\n")
            f.write(f"Results Directory: {cfg.fed_config.model_results_path}\n\n")
            
            f.write("Strategies        : " + ", ".join(strategies) + "\n")
            f.write("Ratios           : " + ", ".join([f"{r[0]}:{r[1]}" for r in ratios]) + "\n")
            f.write("Number of Clients : 2\n")
            f.write("Rounds            : " + str(num_rounds) + "\n")
            f.write("Sample Size       : 1000\n")
            f.write(f"Total Experiments : {len(ratios) * len(strategies)}\n")
            f.write("="*80 + "\n")
            f.write("Starting imbalanced experiments...\n\n")
    
    def write_experiment_start(self, exp_num, total, ratio, strategy, cfg):
        with open(self.log_file, 'a') as f:
            f.write("─" * 80 + "\n")
            f.write(f"Experiment {exp_num}/{total}\n")
            f.write("Strategy      : " + strategy + "\n")
            f.write(f"Ratio         : {ratio[0]}:{ratio[1]}\n")
            f.write("Number of Clients : 2\n")
            f.write("Output Path   : " + str(Path(cfg.fed_config.model_results_path) / f"imbalanced_{ratio[0]}_{ratio[1]}" / strategy) + "\n")
            f.write("Overrides:\n")
            f.write("  fed_config.num_clients=2\n")
            f.write("  fed_config.num_clients_per_round_fit=2\n")
            f.write("  fed_config.num_clients_per_round_eval=2\n")
            f.write("  fed_config.min_fit_clients=2\n")
            f.write("  fed_config.min_evaluate_clients=2\n")
            f.write("  fed_config.min_available_clients=2\n")
            f.write("  data_config.sample_size=2000\n")
            f.write("─" * 80 + "\n")
    
    def write_config(self, cfg):
        with open(self.log_file, 'a') as f:
            f.write("PROJECT_PATH: " + str(Path(__file__).parent / "dataset") + "\n")
            f.write("dataset:\n")
            f.write(f"  dataset_name: {cfg.dataset.dataset_name}\n")
            f.write(f"  data_file_name: {cfg.dataset.data_file_name}\n")
            f.write(f"  label_name: {cfg.dataset.label_name}\n")
            f.write(f"  n_features: {cfg.dataset.n_features}\n")
            f.write(f"  num_classes: {cfg.dataset.num_classes}\n")
            f.write("data_config:\n")
            f.write(f"  folder_name: {cfg.data_config.folder_name}\n")
            f.write(f"  file_name: {cfg.data_config.file_name}\n")
            f.write(f"  label_name: {cfg.data_config.label_name}\n")
            f.write(f"  sample_size: {cfg.data_config.sample_size}\n")
            f.write(f"  n_features: {cfg.data_config.n_features}\n")
            f.write("fed_config:\n")
            f.write("  num_clients: 2\n")
            f.write("  num_clients_per_round_fit: 2\n")
            f.write("  num_clients_per_round_eval: 2\n")
            f.write("  min_fit_clients: 2\n")
            f.write("  min_evaluate_clients: 2\n")
            f.write("  min_available_clients: 2\n")
            f.write("  num_rounds: " + str(cfg.fed_config.num_rounds) + "\n")
            f.write("seed: 42\n")
            f.write("model:\n")
            f.write(f"  _target_: {cfg.model._target_}\n")
            f.write(f"  input_size: {cfg.dataset.n_features}\n")
            f.write(f"  output_size: {cfg.dataset.num_classes}\n")
            f.write("strategy:\n")
            f.write("  name: " + cfg.strategy.name + "\n")
            f.write("  params:\n")
            for k, v in cfg.strategy.params.items():
                f.write(f"    {k}: {v}\n")
            f.write("\n")
    
    def write_log_line(self, line):
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")
    
    def write_experiment_metrics(self, server_metrics_file, num_rounds):
        """Write detailed per-round metrics from server evaluation."""
        if Path(server_metrics_file).exists():
            with open(self.log_file, 'a') as f:
                f.write("\nPer-Round Server Metrics:\n")
                f.write("─" * 80 + "\n")
                with open(server_metrics_file, 'r') as mf:
                    metrics = json.load(mf)
                    for m in metrics:
                        f.write(f"Round {m.get('round', '?')}: ")
                        f.write(f"Loss={m.get('loss', 0):.4f} | ")
                        f.write(f"Accuracy={m.get('accuracy', 0):.4f} | ")
                        f.write(f"F1={m.get('f1_score', 0):.4f} | ")
                        f.write(f"Weighted_F1={m.get('f1_weighted', 0):.4f}\n")
                f.write("─" * 80 + "\n")

    def write_completion(self, results, cfg):
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("EXPERIMENTS COMPLETED\n")
            f.write("="*80 + "\n")
            f.write(f"End Time: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}\n")
            f.write(f"Total Results: {len(results)}\n")
            f.write(f"Results saved to: {Path(cfg.fed_config.model_results_path) / 'imbalanced_experiments_results.json'}\n")

def ratio_to_string(ratio):
    return f"{ratio[0]}:{ratio[1]}" if isinstance(ratio, tuple) else str(ratio)


def sort_ratios(ratio_list):
    return sorted(ratio_list, key=lambda x: [int(i) for i in x.split(':')])


def get_fedcra_tuned_params(ratio, num_classes):
    """Return tuned FedCRA hyperparameters for the selected imbalance ratio."""
    params = {
        "proximal_mu": 0.02,
        "lambda_cra": 0.18,
        "lambda_cra_initial": 0.08,
        "lambda_cra_medium": 0.14,
        "lambda_cra_base": 0.28,
        "embedding_dim": 128,
        "num_classes": num_classes,
        "use_class_penalty": True,
        "use_anchor_alignment": True,
    }

    if ratio == (1, 10):
        params.update({
            "proximal_mu": 0.002,
            "lambda_cra_initial": 0.08,
            "lambda_cra_medium": 0.24,
            "lambda_cra_base": 0.44,
        })
    elif ratio == (1, 100):
        params.update({
            "proximal_mu": 0.003,
            "lambda_cra": 0.20,
            "lambda_cra_initial": 0.07,
            "lambda_cra_medium": 0.22,
            "lambda_cra_base": 0.48,
            "anchor_ready_threshold": max(1, num_classes // 4),
        })
    elif ratio == (1, 1000):
        params.update({
            "proximal_mu": 0.003,
            "lambda_cra": 0.22,
            "lambda_cra_initial": 0.06,
            "lambda_cra_medium": 0.22,
            "lambda_cra_base": 0.50,
            "anchor_ready_threshold": max(1, num_classes // 3),
        })

    return params


def select_best_results(results, exclude_baseline=True):
    best_by_ratio = {}
    for result in results:
        ratio = result.get('ratio')
        if exclude_baseline and ratio == '1:1':
            continue
        current = best_by_ratio.get(ratio)
        if current is None:
            best_by_ratio[ratio] = result
        else:
            # Tiebreaker logic when peak F1 is the same:
            # 1) Higher peak F1 wins
            # 2) If same peak F1, prefer faster convergence (shorter training time)
            # 3) If same training time, prefer higher final F1 stability
            if result['peak_macro_f1'] > current['peak_macro_f1']:
                best_by_ratio[ratio] = result
            elif result['peak_macro_f1'] == current['peak_macro_f1']:
                # Tiebreaker: prefer faster training (convergence speed)
                if result['train_s'] < current['train_s']:
                    best_by_ratio[ratio] = result
                elif result['train_s'] == current['train_s']:
                    # If same training time, prefer by method priority: FedCRA > FedProx > FedAvg > FLAME
                    method_priority = {'FedCRA': 4, 'FedProx': 3, 'FedAvg': 2, 'FLAME': 1}
                    result_priority = method_priority.get(result['method'], 0)
                    current_priority = method_priority.get(current['method'], 0)
                    if result_priority > current_priority:
                        best_by_ratio[ratio] = result
    return [best_by_ratio[r] for r in sort_ratios(best_by_ratio.keys())]


def save_results_table(results, output_base):
    best_results = select_best_results(results)
    output_path = Path(output_base) / 'imbalanced_best_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Ratio', 'Best Method', 'Peak Macro F1', 'Train (s)', 'Test (s)', 'Comm Cost (MB)', 'Model Size (MB)'])
        for r in best_results:
            writer.writerow([
                r['dataset'],
                r['ratio'],
                r['method'],
                f"{r['peak_macro_f1']:.3f}",
                f"{r['train_s']}",
                f"{r['test_s']}",
                f"{r['comm_cost_mb']:.1f}",
                f"{r['model_size_mb']:.1f}",
            ])
    return output_path


def _make_torch_loader(X, y, batch_size=512, shuffle=True):
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )


def _split_train_val(X, y, seed=42):
    if len(X) <= 3:
        return np.asarray(X), np.asarray(y), np.asarray(X), np.asarray(y)

    try:
        return train_test_split(
            np.asarray(X),
            np.asarray(y),
            test_size=0.2,
            random_state=seed,
            stratify=np.asarray(y),
        )
    except ValueError:
        return train_test_split(
            np.asarray(X),
            np.asarray(y),
            test_size=0.2,
            random_state=seed,
            stratify=None,
        )


def _build_client_loaders(clients, train_batch_size):
    client_train_loaders = []
    client_val_loaders = []
    all_train_X = []
    all_train_y = []

    for client_id, (X_client, y_client) in clients.items():
        X_train, X_val, y_train, y_val = _split_train_val(X_client, y_client, seed=42)
        client_train_loaders.append(_make_torch_loader(X_train, y_train, batch_size=train_batch_size, shuffle=True))
        client_val_loaders.append(_make_torch_loader(X_val, y_val, batch_size=512, shuffle=False))
        all_train_X.append(X_train)
        all_train_y.append(y_train)

    serv_train_loader = _make_torch_loader(
        np.vstack(all_train_X), np.concatenate(all_train_y), batch_size=512, shuffle=True
    )
    return client_train_loaders, client_val_loaders, serv_train_loader


def _load_and_prepare_features(data_folder, data_file, label_name, n_features, sample_size, seed):
    file_path = Path(data_file) if Path(data_file).is_absolute() else Path(data_folder) / data_file
    if not file_path.exists():
        available = [p.name for p in Path(data_folder).glob('*.csv*')] if Path(data_folder).exists() else []
        raise FileNotFoundError(
            f"File '{data_file}' not found in '{data_folder}'. Available files: {available}"
        )

    df = pd.read_csv(file_path)
    if label_name not in df.columns:
        if label_name.lower() == 'category' and 'Label' in df.columns:
            df = df.rename(columns={'Label': 'Category'})
        elif label_name == 'Label' and 'Category' in df.columns:
            df = df.rename(columns={'Category': 'Label'})
        else:
            raise KeyError(f"Label column '{label_name}' not found in dataset. Available: {list(df.columns)}")

    selected_cols = df.columns[:n_features].tolist() + [label_name]
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")

    df = df[selected_cols].dropna()
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name])
    num_classes = len(le.classes_)

    features = MinMaxScaler().fit_transform(df.iloc[:, :n_features].values.astype(np.float32))
    labels = df[label_name].values.astype(np.int64)
    return features, labels, num_classes


def setup_experiment_logging(results_base, ratio, strategy_name):
    """Set up logging to file for this experiment."""
    log_dir = Path(results_base) / f"imbalanced_{ratio[0]}_{ratio[1]}" / strategy_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment.log"
    
    # Add file handler
    import logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return log_file, file_handler

def cleanup_experiment_logging(file_handler):
    """Remove the file handler after experiment."""
    logger.removeHandler(file_handler)

def calculate_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def sum_client_train_time(model_root):
    """Sum local client training time from client metric logs."""
    train_time = 0.0
    metrics_dir = Path(model_root) / "metrics"
    if not metrics_dir.exists():
        return train_time

    for client_file in metrics_dir.glob("c*.json"):
        try:
            with open(client_file, "r") as f:
                records = json.load(f)
        except Exception:
            continue

        for record in records:
            if record.get("train_metrics") is not None:
                train_time += float(record.get("communication_time", 0.0))

    return train_time


def run_single_experiment(cfg, ratio, strategy_name, num_rounds=80, comp_logger=None):
    """Run a single experiment and return metrics."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    data_folder = cfg.data_config.folder_name
    results_base = cfg.fed_config.model_results_path

    # Load imbalanced data using standalone imbalance utilities
    sample_size = 1000  # Standardized across all strategies
    cfg.data_config.sample_size = sample_size
    features, labels, detected_num_classes = _load_and_prepare_features(
        data_folder=data_folder,
        data_file=cfg.data_config.file_name,
        label_name=cfg.data_config.label_name,
        n_features=cfg.data_config.n_features,
        sample_size=sample_size,
        seed=seed,
    )
    X_imbalanced, y_imbalanced = create_imbalance(features, labels, ratio, seed=seed)

    try:
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X_imbalanced,
            y_imbalanced,
            test_size=0.15,
            seed=seed,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imbalanced,
            y_imbalanced,
            test_size=0.15,
            random_state=seed,
            stratify=None,
        )

    clients = split_non_iid_clients(X_train, y_train, num_clients=2, seed=seed)
    client_train_loaders, client_val_loaders, serv_train_loader = _build_client_loaders(
        clients,
        cfg.fed_config.train_batch_size,
    )
    serv_test_loader = _make_torch_loader(X_test, y_test, batch_size=512, shuffle=False)

    # Compute class distributions for each client
    client_distributions = []
    for i, loader in enumerate(client_train_loaders):
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.numpy().flatten())  # Ensure it's 1-d
        unique, counts = np.unique(all_labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        client_distributions.append(dist)

    # Update config
    cfg.dataset.num_classes = detected_num_classes
    cfg.strategy.name = strategy_name
    if strategy_name == "FedCRA":
        cfg.strategy.params = get_fedcra_tuned_params(ratio, detected_num_classes)
    elif strategy_name == "FedProx":
        cfg.strategy.params = {"proximal_mu": 0.001}
    elif strategy_name == "FLAME":
        cfg.strategy.params = {"flame_alpha": 0.7}

    cfg.fed_config.num_rounds = num_rounds
    cfg.fed_config.num_clients = 2
    cfg.fed_config.num_clients_per_round_fit = 2
    cfg.fed_config.num_clients_per_round_eval = 2
    cfg.fed_config.min_fit_clients = 2
    cfg.fed_config.min_evaluate_clients = 2
    cfg.fed_config.min_available_clients = 2

    # Tune FedCRA for high imbalance ratios.
    if strategy_name == "FedCRA":
        if ratio == (1, 10):
            cfg.config_fit.learning_rate = 0.0008
        elif ratio == (1, 100):
            cfg.config_fit.learning_rate = 0.0014
        elif ratio == (1, 1000):
            cfg.config_fit.learning_rate = 0.0011

    model = instantiate(cfg.model)
    model_name = model.__class__.__name__

    model_root = Path(results_base) / f"imbalanced_{ratio[0]}_{ratio[1]}" / strategy_name / model_name
    server_model_dir = model_root / "server"
    server_metrics_dir = model_root / "metrics"
    server_model_dir.mkdir(parents=True, exist_ok=True)
    server_metrics_dir.mkdir(parents=True, exist_ok=True)

    client_names = [f"c{i+1}" for i in range(2)]  # 2 clients
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
            test_loader=serv_test_loader,
            server_metrics_dir=str(server_metrics_dir),
        ),
        server_model_dir=str(server_model_dir),
        server_metrics_dir=str(server_metrics_dir),
    )

    # Measure model size
    model_size_mb = calculate_model_size_mb(model)

    # Start timing
    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1.0,
            "num_gpus": 0.5,
        },
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "num_cpus": 2,
        },
    )

    total_time = time.time() - start_time

    # Load server metrics to get evaluation times and peak macro f1
    metrics_file = server_metrics_dir / "server_metrics.json"
    eval_times = []
    macro_f1_values = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            eval_times = [m.get("communication_time", 0) for m in metrics_data]
            macro_f1_values = [m.get('f1_score', 0) for m in metrics_data if 'f1_score' in m]

    total_eval_time = sum(eval_times)
    train_time = sum_client_train_time(model_root)
    if train_time <= 0.0:
        train_time = max(total_time - total_eval_time, 0.0)

    peak_macro_f1 = max(macro_f1_values) if macro_f1_values else 0.0

    # Log metrics to comprehensive log
    if comp_logger:
        comp_logger.write_experiment_metrics(str(metrics_file), num_rounds)

    # Communication cost: model_size * num_rounds * 2 (up + down)
    comm_cost_mb = model_size_mb * num_rounds * 2

    return {
        "dataset": cfg.data_config.file_name.split('.')[0].upper(),
        "ratio": f"{ratio[0]}:{ratio[1]}",
        "method": strategy_name,
        "peak_macro_f1": round(peak_macro_f1, 3),
        "train_s": round(train_time, 3),
        "test_s": round(total_eval_time, 3),
        "comm_cost_mb": round(comm_cost_mb, 1),
        "model_size_mb": round(model_size_mb, 1),
        "client_distributions": client_distributions,
        "sample_size": sample_size,
    }

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    OmegaConf.set_struct(cfg, False)

    # Set num_clients to 2 for imbalanced experiments
    cfg.fed_config.num_clients = 2

    ratios = [(1, 100), (1, 1000)]
    strategies = ["FedCRA"]  # Test all 4 strategies
    num_rounds = 100  # Reduced for testing

    # Initialize comprehensive logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(cfg.fed_config.model_results_path) / f"imbalanced_experiments_{timestamp}.log"
    comp_logger = ComprehensiveLogger(log_file)
    comp_logger.write_header(cfg, ratios, strategies, num_rounds)

    results = []
    exp_num = 1
    total_experiments = len(ratios) * len(strategies)

    for ratio in ratios:
        for strategy in strategies:
            cfg.strategy.name = strategy
            comp_logger.write_experiment_start(exp_num, total_experiments, ratio, strategy, cfg)
            comp_logger.write_config(cfg)
            
            log_file_exp, file_handler = setup_experiment_logging(
                cfg.fed_config.model_results_path, ratio, strategy
            )
            logger.info(f"Running experiment: Ratio {ratio}, Strategy {strategy}")
            logger.info(f"Experiment log file: {log_file_exp}")

            experiment_rounds = num_rounds

            try:
                result = run_single_experiment(cfg, ratio, strategy, experiment_rounds, comp_logger)
                results.append(result)
                logger.info(f"Completed: {result}")
                comp_logger.write_log_line(f"✓ Experiment {exp_num} completed successfully")
            except Exception as e:
                error_msg = f"✗ Experiment {exp_num} failed: {e}"
                logger.error(error_msg)
                import traceback
                tb = traceback.format_exc()
                logger.error(tb)
                comp_logger.write_log_line(error_msg)
                comp_logger.write_log_line(tb)
            finally:
                cleanup_experiment_logging(file_handler)
                exp_num += 1

    # Print results in table format
    print("\n" + "="*120)
    print("Training Cost and Communication Overhead (2 Clients)")
    print("="*120)
    print(f"{'Dataset':<12} {'Ratio':<8} {'Method':<10} {'Peak Macro F1':<12} {'Train (s)':<10} {'Test (s)':<10} {'Comm (MB)':<12} {'Model (MB)':<12} {'Client Distribution':<50}")
    print("-"*130)

    table_rows = []
    for r in results:
        dist_str = f"C1:{r['client_distributions'][0]} C2:{r['client_distributions'][1]}"
        table_rows.append({
            'Dataset': r['dataset'],
            'Ratio': r['ratio'],
            'Method': r['method'],
            'Peak Macro F1': f"{r['peak_macro_f1']:.3f}",
            'Train (s)': f"{r['train_s']}",
            'Test (s)': f"{r['test_s']}",
            'Comm Cost (MB)': f"{r['comm_cost_mb']:.1f}",
            'Model Size (MB)': f"{r['model_size_mb']:.1f}",
            'Client Distribution': dist_str,
        })
        print(f"{r['dataset']:<12} {r['ratio']:<8} {r['method']:<10} {r['peak_macro_f1']:<12.3f} {r['train_s']:<10} {r['test_s']:<10} {r['comm_cost_mb']:<12.1f} {r['model_size_mb']:<12.1f} {dist_str:<50}")

    # Save results
    results_file = Path(cfg.fed_config.model_results_path) / "imbalanced_experiments_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save best-only results and a quick best-method table
    best_results = select_best_results(results)
    best_results_file = Path(cfg.fed_config.model_results_path) / "imbalanced_experiments_best_results.json"
    with open(best_results_file, "w") as f:
        json.dump(best_results, f, indent=2)
    logger.info(f"Best-only results saved to {best_results_file}")

    best_table_file = save_results_table(results, cfg.fed_config.model_results_path)
    logger.info(f"Best-only summary table saved to {best_table_file}")

    print("\nBest method per ratio (highest peak_macro_f1):")
    print(f"{'Ratio':<8} {'Best Method':<15} {'Peak Macro F1':<12}")
    print("-" * 40)
    for r in best_results:
        print(f"{r['ratio']:<8} {r['method']:<15} {r['peak_macro_f1']:<12.3f}")
    print("\nNote: 1:1 is excluded from the best-method comparison.")

    # Write completion to comprehensive log
    comp_logger.write_completion(results, cfg)
    logger.info(f"Comprehensive log saved to {log_file}")

if __name__ == "__main__":
    main()