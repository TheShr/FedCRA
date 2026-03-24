# fed_data.py
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from log_config import base_logger
from src.dataLoaders.data_peprocessing import encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List
import os
import warnings, re

if os.environ.get("SUPPRESS_TORCH_DATALOADER_WARNING", "0") == "1":
    warnings.filterwarnings(
        "ignore",
        message=re.escape("This DataLoader will create") + ".*",
        category=UserWarning,
        module="torch.utils.data.dataloader",
    )

logger = base_logger(__name__)


def choose_num_workers(loaders_per_process: int = 2, hard_cap: int = 8) -> int:
    try:
        cpus_alloc = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    except ValueError:
        cpus_alloc = 0
    cpus = cpus_alloc if cpus_alloc > 0 else (os.cpu_count() or hard_cap)
    cap = min(hard_cap, cpus)
    return max(0, cap // max(1, loaders_per_process))

_NUM_WORKERS = choose_num_workers(loaders_per_process=2)


def get_torch_loader(data, labels, batch_size=1024, shuffle=True):
    tensor_data = TensorDataset(data.cpu(), labels.cpu())
    return DataLoader(
        tensor_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=2 if _NUM_WORKERS > 0 else None,
    )


def split_clients(data_folder, data_file, label_name, n_features=45, num_clients=10, seed=42):
    file_path = Path(data_folder) / data_file
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{data_file}' not found in '{data_folder}'.")

    selected_cols = df.columns[:n_features].tolist() + [label_name]
    df = df[selected_cols]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    grouped = df.groupby(label_name)
    splits = [pd.DataFrame(columns=df.columns) for _ in range(num_clients)]
    for label, group in grouped:
        split_groups = np.array_split(group, num_clients)
        for i, split_group in enumerate(split_groups):
            splits[i] = pd.concat([splits[i], split_group], ignore_index=True)
    for i in range(num_clients):
        splits[i] = splits[i].sample(frac=1, random_state=seed).reset_index(drop=True)
    return splits


def federated_data(data_folder, data_file, label_name, n_features, num_clients=10,
                   train_batch_size=64):
    client_train_loaders = []
    client_val_loaders = []
    train_data, train_labels, test_data, test_labels = [], [], [], []

    client_splits = split_clients(data_folder=data_folder, data_file=data_file,
                                  label_name=label_name, n_features=n_features,
                                  num_clients=num_clients)
    for index, c_df in enumerate(client_splits):
        le = LabelEncoder()
        c_df[label_name] = le.fit_transform(c_df[label_name])
        logger.info(f"Client {index + 1} - Class Labels Mapping: "
                    f"{dict(zip(le.classes_, le.transform(le.classes_)))}")
        c_data = MinMaxScaler().fit_transform(c_df.iloc[:, :n_features])
        c_targets = c_df[label_name]
        X_train, X_test, y_train, y_test = train_test_split(c_data, c_targets, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test])
        y_train, y_val, y_test = map(lambda y: torch.tensor(y.values, dtype=torch.long), [y_train, y_val, y_test])
        client_train_loaders.append(get_torch_loader(X_train, y_train, batch_size=train_batch_size))
        client_val_loaders.append(get_torch_loader(X_val, y_val, batch_size=512))
        train_data.append(X_train); train_labels.append(y_train)
        test_data.append(X_test); test_labels.append(y_test)

    train_data, train_labels = torch.cat(train_data), torch.cat(train_labels)
    test_data, test_labels   = torch.cat(test_data),  torch.cat(test_labels)
    return (client_train_loaders, client_val_loaders,
            get_torch_loader(train_data, train_labels, batch_size=512),
            get_torch_loader(test_data, test_labels, batch_size=512))


# ============================================================
# Dirichlet non-IID federated data split  (FedCRA / FedAvg)
# ============================================================
def federated_data_dirichlet(
        data_folder: str,
        data_file: str,
        label_name: str,
        n_features: int,
        num_clients: int = 8,
        train_batch_size: int = 128,
        alpha: float = 0.1,
        sample_size: int = 8000,
        seed: int = 42,
        held_out_frac: float = 0.15,   # fraction carved out for balanced global test set
):
    """
    Dirichlet-based non-IID split with a STRATIFIED held-out test set.

    Key design:
      1. Carve held_out_frac of each class BEFORE the Dirichlet split.
         This gives a balanced global test set for fair macro-metric evaluation.
      2. Apply Dirichlet(alpha) only to the remaining training pool.
      3. Each client gets 80/10/10 train/val from its own shard.
      4. Global test = the stratified held-out set (untouched by Dirichlet).

    Fixed seed => identical distribution across FedAvg and FedCRA runs.
    """
    rng = np.random.default_rng(seed)

    file_path = Path(data_folder) / data_file
    logger.info(f"[Dirichlet] Loading data from {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(
            f"File '{data_file}' not found in '{data_folder}'.\n"
            f"Full path tried: {file_path}"
        )
    df = pd.read_csv(file_path)

    selected_cols = df.columns[:n_features].tolist() + [label_name]
    df = df[selected_cols].dropna()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Global label encoding — same mapping every run
    le = LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name])
    logger.info(f"[Dirichlet] Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    num_classes = len(le.classes_)

    # Scale features globally
    features = MinMaxScaler().fit_transform(
        df.iloc[:, :n_features].values.astype(np.float32))
    labels = df[label_name].values.astype(np.int64)

    # ── Step 1: Carve stratified held-out test set ──────────────────────────
    # For each class, reserve held_out_frac of samples for the global test set.
    # The rest goes into the Dirichlet training pool.
    held_X, held_y = [], []
    train_pool_indices = []

    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)
        n_held = max(1, int(len(class_idx) * held_out_frac))
        held_X.append(features[class_idx[:n_held]])
        held_y.append(labels[class_idx[:n_held]])
        train_pool_indices.extend(class_idx[n_held:].tolist())

    # Global balanced test loader
    held_X_t = torch.tensor(np.vstack(held_X), dtype=torch.float32)
    held_y_t = torch.tensor(np.concatenate(held_y), dtype=torch.long)
    serv_test = get_torch_loader(held_X_t, held_y_t, batch_size=512, shuffle=False)
    logger.info(f"[Dirichlet] Held-out test set: {len(held_y_t)} samples "
                f"(balanced across {num_classes} classes)")

    # ── Step 2: Dirichlet allocation over training pool only ────────────────
    pool_features = features[train_pool_indices]
    pool_labels   = labels[train_pool_indices]

    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        class_idx = np.where(pool_labels == k)[0]
        if len(class_idx) == 0:
            continue
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(class_idx)).astype(int)
        remainder = len(class_idx) - counts.sum()
        if remainder > 0:
            for c in rng.integers(0, num_clients, size=remainder):
                counts[c] += 1
        ptr = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(class_idx[ptr: ptr + cnt].tolist())
            ptr += cnt

    # ── Step 3: Per-client loaders ──────────────────────────────────────────
    client_train_loaders, client_val_loaders = [], []

    for i, idx in enumerate(client_indices):
        if len(idx) < 10:
            logger.warning(f"[Dirichlet] Client {i+1} has {len(idx)} samples — padding to 10.")
            extra = rng.integers(0, len(pool_features), size=10 - len(idx)).tolist()
            idx = idx + extra

        idx = np.array(idx)
        rng.shuffle(idx)
        X = torch.tensor(pool_features[idx], dtype=torch.float32)
        y = torch.tensor(pool_labels[idx], dtype=torch.long)

        n = len(X)
        n_train = max(1, int(n * 0.8))
        n_val   = max(1, int(n * 0.1))

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]

        unique, cnts = torch.unique(y_train, return_counts=True)
        logger.info(f"[Dirichlet] Client {i+1}: {n_train} train | "
                    f"classes={unique.tolist()} counts={cnts.tolist()}")

        client_train_loaders.append(
            get_torch_loader(X_train, y_train, batch_size=train_batch_size))
        client_val_loaders.append(
            get_torch_loader(X_val, y_val, batch_size=512))

    # ── Step 4: Global server train loader (full pool, balanced shuffle) ────
    pool_X_t = torch.tensor(pool_features, dtype=torch.float32)
    pool_y_t = torch.tensor(pool_labels,   dtype=torch.long)
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm  = torch.randperm(len(pool_X_t), generator=gen)
    split = int(len(pool_X_t) * 0.8)
    serv_train = get_torch_loader(
        pool_X_t[perm[:split]], pool_y_t[perm[:split]], batch_size=512)

    return client_train_loaders, client_val_loaders, serv_train, serv_test
