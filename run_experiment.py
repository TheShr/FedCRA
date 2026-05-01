#!/usr/bin/env python3
"""Run a single federated learning experiment from a YAML spec."""

import sys

import numpy as np
try:
    from hydra import compose, initialize
except ImportError:
    from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

import main_fed
from src.fedLearn.fed_data import load_raw_dataset


def parse_custom_imbalance(imbalance_ratio: str, labels):
    ratio_str = str(imbalance_ratio).strip()
    if ':' not in ratio_str:
        raise ValueError("custom_imbalance must be a string like '1:10' or '1:100'.")
    low, high = ratio_str.split(':', 1)
    try:
        low_val = float(low)
        high_val = float(high)
    except ValueError as exc:
        raise ValueError("custom_imbalance values must be numeric") from exc
    if low_val <= 0 or high_val <= 0:
        raise ValueError("custom_imbalance values must be positive")

    min_ratio, max_ratio = min(low_val, high_val), max(low_val, high_val)
    num_classes = len(labels)
    if num_classes == 1:
        return {labels[0]: 1.0}

    ratios = np.geomspace(max_ratio, min_ratio, num=num_classes)
    ratios = ratios.tolist()
    sorted_labels = sorted(labels)
    return {label: float(val) for label, val in zip(sorted_labels, ratios)}


def resolve_experiment_config(config_name: str = "custom_experiment"):
    with initialize(config_path="conf", job_name="run_custom_experiment"):
        cfg = compose(config_name=config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    return cfg


def apply_imbalance_overrides(cfg):
    if cfg.get("custom_imbalance"):
        raw_df = load_raw_dataset(
            data_folder=cfg.data_config.folder_name,
            data_file=cfg.data_config.file_name,
            label_name=cfg.data_config.label_name,
            n_features=cfg.data_config.n_features,
            seed=cfg.seed,
        )
        labels = sorted(raw_df[cfg.data_config.label_name].unique().tolist())
        cfg.imbalance.class_ratios = parse_custom_imbalance(cfg.custom_imbalance, labels)


def run_experiment(config_name: str = "custom_experiment"):
    cfg = resolve_experiment_config(config_name)
    apply_imbalance_overrides(cfg)
    print(OmegaConf.to_yaml(cfg))
    main_fed.main.__wrapped__(cfg)


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "custom_experiment"
    run_experiment(config_name)
