# fed_data.py
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from log_config import base_logger
from src.dataLoaders.data_peprocessing import encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Optional
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

_NUM_WORKERS = 0  # Disabled multiprocessing to avoid fork errors in constrained environments


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


def load_raw_dataset(data_folder: str, data_file: str, label_name: str, n_features: int, seed: int = 42) -> pd.DataFrame:
    file_path = Path(data_file) if Path(data_file).is_absolute() else Path(data_folder) / data_file
    if not file_path.exists():
        available = [p.name for p in Path(data_folder).glob('*.csv*')] if Path(data_folder).exists() else []
        raise FileNotFoundError(
            f"File '{data_file}' not found in '{data_folder}'. Available files: {available}"
        )

    if str(file_path).endswith('.bz2'):
        import bz2
        with bz2.open(file_path, 'rt') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(file_path)

    if label_name not in df.columns:
        if label_name.lower() == 'category' and 'Label' in df.columns:
            df = df.rename(columns={'Label': 'Category'})
        elif label_name == 'Label' and 'Category' in df.columns:
            df = df.rename(columns={'Category': 'Label'})
        else:
            raise KeyError(f"Label column '{label_name}' not found in dataset. Available: {list(df.columns)}")

    selected_cols = df.columns[:n_features].tolist() + [label_name]
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df[selected_cols].dropna()
    if df.empty:
        raise ValueError('Loaded dataset is empty after dropping NaN values')

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def _normalize_class_ratios(class_ratios: dict) -> dict:
    ratios = {cls: float(value) for cls, value in class_ratios.items()}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError('class_ratios must sum to a positive value')
    return {cls: ratio / total for cls, ratio in ratios.items()}


def _match_class_ratios_to_present_classes(class_ratios: dict, present_classes: set) -> tuple[dict, list]:
    normalized_present = {_normalize_label(cls): cls for cls in present_classes}
    matched = {}
    missing = []
    for cls, ratio in class_ratios.items():
        normalized = _normalize_label(cls)
        if normalized in normalized_present:
            matched[normalized_present[normalized]] = float(ratio)
        else:
            missing.append(cls)
    return matched, sorted(missing)


def _target_counts_from_ratios(class_ratios: dict, total_samples: int) -> dict:
    normalized = _normalize_class_ratios(class_ratios)
    raw_counts = {cls: normalized[cls] * total_samples for cls in normalized}
    floored = {cls: int(np.floor(count)) for cls, count in raw_counts.items()}
    remainder = total_samples - sum(floored.values())
    if remainder > 0:
        ranked = sorted(raw_counts.keys(), key=lambda cls: raw_counts[cls] - floored[cls], reverse=True)
        for cls in ranked[:remainder]:
            floored[cls] += 1
    return floored


def create_controlled_dataset(
        df: pd.DataFrame,
        class_ratios: dict,
        total_samples: int,
        method: str = 'oversample',
        label_name: str = 'Category',
        seed: int = 42,
) -> pd.DataFrame:
    if label_name not in df.columns:
        raise KeyError(f"Label column '{label_name}' not found in dataset")
    if total_samples <= 0:
        raise ValueError('total_samples must be positive')
    if method != 'oversample':
        raise ValueError("Unsupported method, only 'oversample' is supported")

    present_classes = set(df[label_name].astype(str).unique())
    requested_classes = {cls: float(count) for cls, count in class_ratios.items()}
    matched_classes, missing = _match_class_ratios_to_present_classes(requested_classes, present_classes)

    logger.info(
        'create_controlled_dataset: present classes=%s, requested classes=%s, matched classes=%s',
        sorted(present_classes), sorted(requested_classes.keys()), sorted(matched_classes.keys()),
    )

    if missing:
        logger.warning(
            'create_controlled_dataset: ignoring missing class labels not present in dataset: %s',
            missing,
        )

    if not matched_classes:
        matched_classes = {cls: 1.0 for cls in present_classes}
        logger.info(
            'create_controlled_dataset: no requested class labels found in dataset, falling back to uniform ratio over %d present classes.',
            len(present_classes),
        )

    target_counts = _target_counts_from_ratios(matched_classes, total_samples)
    rng = np.random.RandomState(seed)
    selected_frames = []
    for cls, count in target_counts.items():
        if count == 0:
            continue
        class_indices = df.index[df[label_name].astype(str) == str(cls)].to_numpy()
        if class_indices.size == 0:
            raise ValueError(f"No available samples for class '{cls}'")
        replace = len(class_indices) < count
        sample_indices = rng.choice(class_indices, size=count, replace=replace)
        selected_frames.append(df.loc[sample_indices])

    controlled_df = pd.concat(selected_frames, ignore_index=True)
    controlled_df = controlled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(controlled_df) != total_samples:
        raise AssertionError('Controlled dataset size mismatch')
    return controlled_df


def split_clients(data_folder, data_file, label_name, n_features=45, num_clients=10, seed=42):
    file_path = Path(data_file) if Path(data_file).is_absolute() else Path(data_folder) / data_file
    logger.info(f"Loading data from {file_path}")
    try:
        # Handle BZ2 compressed files
        if str(file_path).endswith('.bz2'):
            import bz2
            with bz2.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
    except FileNotFoundError:
        available = [p.name for p in Path(data_folder).glob('*.csv*')] if Path(data_folder).exists() else []
        raise FileNotFoundError(
            f"File '{data_file}' not found in '{data_folder}'. Available files: {available}"
        )
    except Exception as e:
        raise RuntimeError(f"Error reading {data_file}: {e}")
    
    logger.info(f"Available columns: {list(df.columns)}")
    logger.info(f"Looking for label column: '{label_name}'")
    
    if label_name not in df.columns:
        if label_name.lower() == 'category' and 'Label' in df.columns:
            df = df.rename(columns={'Label': 'Category'})
            logger.info("Renamed 'Label' column to 'Category' for dataset compatibility.")
        elif label_name == 'Label' and 'Category' in df.columns:
            df = df.rename(columns={'Category': 'Label'})
            logger.info("Renamed 'Category' column to 'Label' for dataset compatibility.")
        else:
            raise KeyError(f"Label column '{label_name}' not found. Available: {list(df.columns)}")

    selected_cols = df.columns[:n_features].tolist() + [label_name]
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    
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
            get_torch_loader(test_data, test_labels, batch_size=512),
            num_classes)


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
        min_client_samples: int = 100,
        total_samples: int = 100000,
        class_ratios: Optional[dict] = None,
        seed: int = 42,
        held_out_frac: float = 0.15,
):
    """
    Dirichlet-based non-IID federated split with:
      - Controlled total dataset size and imbalance profile
      - Stratified held-out global test set
      - Minimum client size enforced
      - Controlled alpha for moderate skew
    """
    rng = np.random.default_rng(seed)
    df = load_raw_dataset(data_folder, data_file, label_name, n_features, seed=seed)

    if class_ratios is not None:
        logger.info('Applying controlled dataset creation with requested class ratios: %s', list(class_ratios.keys()))
        df = create_controlled_dataset(
            df,
            class_ratios=class_ratios,
            total_samples=total_samples,
            method='oversample',
            label_name=label_name,
            seed=seed,
        )
    elif total_samples and len(df) > total_samples:
        df = df.sample(n=total_samples, random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name])
    class_names = list(le.classes_)
    num_classes = len(class_names)
    logger.info('federated_data_dirichlet: final encoded class_names=%s', class_names)

    features = MinMaxScaler().fit_transform(df.iloc[:, :n_features].values.astype(np.float32))
    labels = df[label_name].values.astype(np.int64)

    held_X, held_y = [], []
    train_pool_indices = []

    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)
        n_held = max(1, int(len(class_idx) * held_out_frac))
        held_X.append(features[class_idx[:n_held]])
        held_y.append(labels[class_idx[:n_held]])
        train_pool_indices.extend(class_idx[n_held:].tolist())

    held_X_t = torch.tensor(np.vstack(held_X), dtype=torch.float32)
    held_y_t = torch.tensor(np.concatenate(held_y), dtype=torch.long)
    serv_test = get_torch_loader(held_X_t, held_y_t, batch_size=512, shuffle=False)

    pool_features = features[train_pool_indices]
    pool_labels = labels[train_pool_indices]
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

    for i in range(num_clients):
        if len(client_indices[i]) < min_client_samples:
            deficit = min_client_samples - len(client_indices[i])
            largest = np.argmax([len(c) for c in client_indices])
            extra = rng.choice(client_indices[largest], size=deficit, replace=False).tolist()
            client_indices[i].extend(extra)
            client_indices[largest] = list(set(client_indices[largest]) - set(extra))
            rng.shuffle(client_indices[i])
            rng.shuffle(client_indices[largest])

    client_train_loaders, client_val_loaders = [], []

    for i, idx in enumerate(client_indices):
        idx = np.array(idx)
        rng.shuffle(idx)

        X = torch.tensor(pool_features[idx], dtype=torch.float32)
        y = torch.tensor(pool_labels[idx], dtype=torch.long)

        unique, counts = np.unique(pool_labels[idx], return_counts=True)
        dist = dict(zip(unique.tolist(), (counts / counts.sum()).round(3).tolist()))
        logger.info(f"[Dirichlet] Client {i+1}: class dist = {dist}")

        n = len(X)
        n_train = max(int(n * 0.8), 1)
        n_val = max(int(n * 0.1), 10)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]

        client_train_loaders.append(get_torch_loader(X_train, y_train, batch_size=train_batch_size))
        client_val_loaders.append(get_torch_loader(X_val, y_val, batch_size=512))

        logger.info(f"[Dirichlet] Client {i+1}: {len(X_train)} train | "
                    f"{len(X_val)} val | classes={torch.unique(y_train).tolist()}")

    pool_X_t = torch.tensor(pool_features, dtype=torch.float32)
    pool_y_t = torch.tensor(pool_labels, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pool_X_t), generator=gen)
    split = int(len(pool_X_t) * 0.8)
    serv_train = get_torch_loader(pool_X_t[perm[:split]], pool_y_t[perm[:split]], batch_size=512)

    return client_train_loaders, client_val_loaders, serv_train, serv_test, num_classes, class_names


def federated_data_imbalanced(
        data_folder: str,
        data_file: str,
        label_name: str,
        n_features: int,
        ratio: tuple = (1, 10),
        train_batch_size: int = 128,
        sample_size: int = 8000,
        seed: int = 42,
        held_out_frac: float = 0.15,
):
    """
    Imbalanced federated split for 2 clients with specified ratio.
    """
    from src.imbalanced_split import create_two_client_imbalance
    from torch.utils.data import TensorDataset

    rng = np.random.default_rng(seed)
    file_path = Path(data_file) if Path(data_file).is_absolute() else Path(data_folder) / data_file
    if not file_path.exists():
        available = [p.name for p in Path(data_folder).glob('*.csv*')] if Path(data_folder).exists() else []
        raise FileNotFoundError(
            f"File '{data_file}' not found in '{data_folder}'. Available files: {available}"
        )

    df = pd.read_csv(file_path)
    
    # Debug: Check available columns
    logger.info(f"Available columns in {file_path.name}: {list(df.columns)}")
    logger.info(f"Looking for label column: '{label_name}'")
    
    # Validate that label column exists
    if label_name not in df.columns:
        if label_name.lower() == 'category' and 'Label' in df.columns:
            df = df.rename(columns={'Label': 'Category'})
            logger.info("Renamed 'Label' column to 'Category' for dataset compatibility.")
        elif label_name == 'Label' and 'Category' in df.columns:
            df = df.rename(columns={'Category': 'Label'})
            logger.info("Renamed 'Category' column to 'Label' for dataset compatibility.")
        else:
            raise KeyError(f"Label column '{label_name}' not found in dataset. Available columns: {list(df.columns)}")
    
    selected_cols = df.columns[:n_features].tolist() + [label_name]
    logger.info(f"Selected columns (first {n_features} + label): {selected_cols}")
    
    # Ensure all selected columns exist
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")
    
    df = df[selected_cols].dropna()
    logger.info(f"Data shape after column selection and NaN removal: {df.shape}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Global label encoding
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    le = LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name])
    num_classes = len(le.classes_)

    # Scale features globally
    features = MinMaxScaler().fit_transform(df.iloc[:, :n_features].values.astype(np.float32))
    labels = df[label_name].values.astype(np.int64)

    # ── Step 1: Stratified held-out test set ──────────────────────────
    held_X, held_y = [], []
    train_pool_indices = []

    for k in range(num_classes):
        class_idx = np.where(labels == k)[0]
        rng.shuffle(class_idx)
        n_held = max(1, int(len(class_idx) * held_out_frac))
        held_X.append(features[class_idx[:n_held]])
        held_y.append(labels[class_idx[:n_held]])
        train_pool_indices.extend(class_idx[n_held:].tolist())

    held_X_t = torch.tensor(np.vstack(held_X), dtype=torch.float32)
    held_y_t = torch.tensor(np.concatenate(held_y), dtype=torch.long)
    serv_test = get_torch_loader(held_X_t, held_y_t, batch_size=512, shuffle=False)

    # ── Step 2: Imbalanced split for 2 clients ───────────────────────
    pool_features = features[train_pool_indices]
    pool_labels = labels[train_pool_indices]

    # Create TensorDataset for the pool
    pool_dataset = TensorDataset(
        torch.tensor(pool_features, dtype=torch.float32),
        torch.tensor(pool_labels, dtype=torch.long)
    )

    # Use imbalanced_split
    client_1_data, client_2_data = create_two_client_imbalance(
        pool_dataset, ratio=ratio, shuffle=True, seed=seed
    )

    # ── Step 3: Per-client train/val loaders ───────────────────────
    client_train_loaders, client_val_loaders = [], []

    for i, client_data in enumerate([client_1_data, client_2_data]):
        # Split into train/val
        n = len(client_data)
        if n <= 10:
            # For very small clients, use all for training, duplicate one sample for val
            n_train = n
            n_val = 1
            train_indices = list(range(n))
            val_indices = [0]  # Duplicate first sample
        else:
            n_train = max(int(n * 0.8), 1)
            n_val = max(int(n * 0.1), 1)
            indices = list(range(n))
            rng.shuffle(indices)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]

        # Create subsets
        train_subset = Subset(client_data, train_indices)
        val_subset = Subset(client_data, val_indices)

        # Create loaders
        client_train_loaders.append(DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True,
            num_workers=_NUM_WORKERS, pin_memory=True,
            persistent_workers=(_NUM_WORKERS > 0),
            prefetch_factor=2 if _NUM_WORKERS > 0 else None,
        ))
        client_val_loaders.append(DataLoader(
            val_subset, batch_size=512, shuffle=False,
            num_workers=_NUM_WORKERS, pin_memory=True,
            persistent_workers=(_NUM_WORKERS > 0),
            prefetch_factor=2 if _NUM_WORKERS > 0 else None,
        ))

        # Log class distribution
        all_labels = []
        for _, labels_batch in train_subset:
            all_labels.append(labels_batch.item())  # labels_batch is 0-d tensor
        unique, counts = np.unique(all_labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"[Imbalanced] Client {i+1}: {len(train_subset)} train | "
                    f"{len(val_subset)} val | classes={dist}")

    # ── Step 4: Global server train loader ───────────────────────
    pool_X_t = torch.tensor(pool_features, dtype=torch.float32)
    pool_y_t = torch.tensor(pool_labels, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pool_X_t), generator=gen)
    split = int(len(pool_X_t) * 0.8)
    serv_train = get_torch_loader(pool_X_t[perm[:split]], pool_y_t[perm[:split]], batch_size=512)

    return client_train_loaders, client_val_loaders, serv_train, serv_test, num_classes