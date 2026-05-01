"""Federated data imbalance utilities.

This module provides a standalone imbalance and non-IID client data pipeline
that can be imported and used by existing federated training scripts without
changing the training loops or core pipelines.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VALID_RATIOS = {'1:1': (1, 1), '1:10': (1, 10), '1:100': (1, 100), '1:1000': (1, 1000)}


def _to_numpy_array(data: Any) -> np.ndarray:
    return np.asarray(data)


def _print_distribution(label: str, y: np.ndarray) -> None:
    counter = Counter(y.tolist())
    distribution = ', '.join(f'{cls}:{count}' for cls, count in sorted(counter.items()))
    print(f'{label} distribution -> {distribution}')


def _validate_ratio(ratio: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(ratio, tuple):
        if ratio in VALID_RATIOS.values():
            return ratio
        raise ValueError(f'Unsupported ratio tuple: {ratio}. Supported ratios: {list(VALID_RATIOS.keys())}')

    ratio_str = str(ratio).strip()
    if ratio_str not in VALID_RATIOS:
        raise ValueError(f'Unsupported ratio: {ratio}. Supported ratios: {list(VALID_RATIOS.keys())}')
    return VALID_RATIOS[ratio_str]


def create_imbalance(
    X: Any,
    y: Any,
    ratio: Union[str, Tuple[int, int]],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an imbalanced dataset from X, y using a target minority:majority ratio.

    The function undersamples majority classes and optionally oversamples the
    minority class when necessary. It supports ratios 1:1, 1:10, 1:100, and
    1:1000.
    """
    X_arr = _to_numpy_array(X)
    y_arr = _to_numpy_array(y)

    if len(X_arr) != len(y_arr):
        raise ValueError('X and y must have the same length')

    ratio_num, ratio_den = _validate_ratio(ratio)
    counts = Counter(y_arr.tolist())
    if len(counts) == 0:
        return X_arr, y_arr

    _print_distribution('Global before imbalance', y_arr)

    classes = sorted(counts.keys())
    class_counts = {cls: counts[cls] for cls in classes}
    minority_label = min(class_counts, key=lambda cls: (class_counts[cls], cls))
    majority_labels = [cls for cls in classes if cls != minority_label]

    minority_count = class_counts[minority_label]
    if ratio_num == ratio_den == 1:
        target_count = minority_count
    else:
        target_count = int(minority_count * ratio_den / ratio_num)
        target_count = max(target_count, 1)

    rng = np.random.RandomState(seed)
    selected_indices: List[int] = []

    if ratio_num == ratio_den == 1:
        for cls in classes:
            cls_indices = np.flatnonzero(y_arr == cls)
            if len(cls_indices) <= target_count:
                selected = cls_indices
            else:
                selected = rng.choice(cls_indices, size=target_count, replace=False)
            selected_indices.extend(selected.tolist())
    else:
        available_majority = [class_counts[cls] for cls in majority_labels]
        if available_majority and min(available_majority) < target_count:
            target_count = min(available_majority)
            required_minority = int(round(target_count * ratio_num / ratio_den))
            if required_minority > minority_count:
                minority_count = required_minority

        for cls in classes:
            cls_indices = np.flatnonzero(y_arr == cls)
            if cls == minority_label:
                if len(cls_indices) >= minority_count:
                    selected = rng.choice(cls_indices, size=minority_count, replace=False)
                else:
                    selected = rng.choice(cls_indices, size=minority_count, replace=True)
            else:
                if len(cls_indices) <= target_count:
                    selected = cls_indices
                else:
                    selected = rng.choice(cls_indices, size=target_count, replace=False)
            selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)
    X_imbalanced = X_arr[selected_indices]
    y_imbalanced = y_arr[selected_indices]

    _print_distribution('Global after imbalance', y_imbalanced)
    return X_imbalanced, y_imbalanced


def train_test_split_stratified(
    X: Any,
    y: Any,
    test_size: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets while preserving the class imbalance."""
    X_arr = _to_numpy_array(X)
    y_arr = _to_numpy_array(y)

    if len(X_arr) != len(y_arr):
        raise ValueError('X and y must have the same length')
    if len(X_arr) == 0:
        return X_arr, X_arr, y_arr, y_arr

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=seed,
    )

    _print_distribution('Train set', y_train)
    _print_distribution('Test set', y_test)
    return X_train, X_test, y_train, y_test


def _normalize_class_ratios(class_ratios: Dict[Union[str, int], float]) -> Dict[Union[str, int], float]:
    if class_ratios is None:
        raise ValueError('class_ratios must be provided')
    ratios = {cls: float(value) for cls, value in class_ratios.items()}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError('class_ratios must sum to a positive value')
    return {cls: ratio / total for cls, ratio in ratios.items()}


def _target_counts_from_ratios(class_ratios: Dict[Union[str, int], float], total_samples: int) -> Dict[Union[str, int], int]:
    normalized = _normalize_class_ratios(class_ratios)
    raw_counts = {cls: normalized[cls] * total_samples for cls in normalized}
    floored_counts = {cls: int(np.floor(count)) for cls, count in raw_counts.items()}
    remainder = total_samples - sum(floored_counts.values())
    if remainder > 0:
        ranked = sorted(raw_counts.keys(), key=lambda cls: raw_counts[cls] - floored_counts[cls], reverse=True)
        for cls in ranked[:remainder]:
            floored_counts[cls] += 1
    return floored_counts


def create_controlled_dataset(
    dataset: Any,
    class_ratios: Dict[Union[str, int], float],
    total_samples: int,
    method: str = 'oversample',
    label_name: Optional[str] = None,
    seed: int = 42,
):
    """Create a controlled dataset with exact total_samples and a configurable class ratio.

    The function preserves class labels and applies random oversampling when a class
    has fewer examples than the target count. When a class has more examples than the
    target, it selects a light random sample without replacement.
    """
    if total_samples <= 0:
        raise ValueError('total_samples must be positive')
    if method not in {'oversample'}:
        raise ValueError("Unsupported method, only 'oversample' is supported")

    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
        if label_name is None:
            label_name = df.columns[-1]
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        X_arr = _to_numpy_array(dataset[0])
        y_arr = _to_numpy_array(dataset[1])
        df = pd.DataFrame(X_arr)
        df['label'] = y_arr
        label_name = label_name or 'label'
    else:
        raise ValueError('dataset must be a pandas DataFrame or a tuple/list of (X, y)')

    if label_name not in df.columns:
        raise KeyError(f"Label column '{label_name}' not found in dataset")

    if not df.shape[0]:
        raise ValueError('Input dataset is empty')

    requested_classes = {cls: float(value) for cls, value in class_ratios.items()}
    present_classes = set(df[label_name].unique())
    matched_classes = {cls: ratio for cls, ratio in requested_classes.items() if cls in present_classes}

    if not matched_classes:
        if not present_classes:
            raise ValueError('Input dataset contains no classes')
        matched_classes = {cls: 1.0 for cls in present_classes}
        warnings.warn(
            'create_controlled_dataset: no requested class labels found in dataset; falling back to uniform ratio over present classes.',
            UserWarning,
        )
    elif len(matched_classes) < len(requested_classes):
        missing_classes = sorted(set(requested_classes) - set(matched_classes))
        warnings.warn(
            f'create_controlled_dataset: ignoring missing class labels not present in dataset: {missing_classes}',
            UserWarning,
        )

    target_counts = _target_counts_from_ratios(matched_classes, total_samples)
    rng = np.random.RandomState(seed)
    selected_frames = []

    for cls, target_count in target_counts.items():
        cls_indices = df.index[df[label_name] == cls].to_numpy()
        if target_count == 0:
            continue
        if len(cls_indices) == 0:
            raise ValueError(f"No available samples for class '{cls}' to satisfy target count")
        if len(cls_indices) >= target_count:
            selected_idxs = rng.choice(cls_indices, size=target_count, replace=False)
        else:
            selected_idxs = rng.choice(cls_indices, size=target_count, replace=True)
        selected_frames.append(df.loc[selected_idxs])

    controlled_df = pd.concat(selected_frames, ignore_index=True)
    controlled_df = controlled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(controlled_df) != total_samples:
        raise AssertionError('Controlled dataset size mismatch')

    if isinstance(dataset, pd.DataFrame):
        return controlled_df

    X_out = controlled_df.drop(columns=[label_name]).to_numpy()
    y_out = controlled_df[label_name].to_numpy()
    return X_out, y_out


def specialized_client_split(
    dataset: Any,
    label_name: str = 'Category',
    num_clients: int = 5,
    seed: int = 42,
):
    """Create a 5-client specialized partition with per-client class preferences.

    Clients:
        C1 -> Normal + DoS
        C2 -> mostly Normal
        C3 -> mixed
        C4 -> 80% ARP
        C5 -> 80% Recon

    The function preserves an exact total dataset size across clients and keeps a
    minimum mixed share for each client.
    """
    if num_clients != 5:
        raise ValueError('specialized_client_split currently supports exactly 5 clients')
    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        X_arr = _to_numpy_array(dataset[0])
        y_arr = _to_numpy_array(dataset[1])
        df = pd.DataFrame(X_arr)
        df[label_name] = y_arr
    else:
        raise ValueError('dataset must be a pandas DataFrame or a tuple/list of (X, y)')

    if label_name not in df.columns:
        raise KeyError(f"Label column '{label_name}' not found in dataset")

    total_samples = len(df)
    if total_samples < num_clients:
        raise ValueError('Dataset must contain at least one sample per client')

    client_size = total_samples // num_clients
    remainders = total_samples - client_size * num_clients
    client_sizes = [client_size + (1 if i < remainders else 0) for i in range(num_clients)]

    client_profiles = [
        {'name': 'C1', 'core_classes': ['Normal', 'DoS'], 'core_frac': 0.8},
        {'name': 'C2', 'core_classes': ['Normal'], 'core_frac': 0.8},
        {'name': 'C3', 'core_classes': [], 'core_frac': 0.0},
        {'name': 'C4', 'core_classes': ['ARP'], 'core_frac': 0.8},
        {'name': 'C5', 'core_classes': ['Recon'], 'core_frac': 0.8},
    ]

    remaining_pool = df.copy()
    rng = np.random.RandomState(seed)
    clients: Dict[str, pd.DataFrame] = {}

    for idx, profile in enumerate(client_profiles):
        size = client_sizes[idx]
        core_target = int(round(size * profile['core_frac']))
        mixed_target = size - core_target
        core_sample = pd.DataFrame(columns=df.columns)

        if core_target > 0 and profile['core_classes']:
            core_candidates = remaining_pool[remaining_pool[label_name].isin(profile['core_classes'])]
            if len(core_candidates) >= core_target:
                core_sample = core_candidates.sample(n=core_target, random_state=seed + idx, replace=False)
            else:
                core_sample = core_candidates.sample(n=core_target, random_state=seed + idx, replace=True)

        mixed_candidates = remaining_pool.drop(index=core_sample.index, errors='ignore')
        if len(mixed_candidates) >= mixed_target:
            mixed_sample = mixed_candidates.sample(n=mixed_target, random_state=seed + idx + 10, replace=False)
            remaining_pool = mixed_candidates.drop(index=mixed_sample.index, errors='ignore')
        else:
            mixed_sample = mixed_candidates
            if len(mixed_sample) < mixed_target:
                extra = df.sample(n=mixed_target - len(mixed_sample), random_state=seed + idx + 20, replace=True)
                mixed_sample = pd.concat([mixed_sample, extra], ignore_index=True)

        client_df = pd.concat([core_sample, mixed_sample], ignore_index=True)
        client_df = client_df.sample(frac=1, random_state=seed + idx + 30).reset_index(drop=True)
        clients[profile['name']] = client_df

    return clients


def split_non_iid_clients(
    X: Any,
    y: Any,
    num_clients: int = 10,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create a realistic non-IID client distribution for federated experiments."""
    X_arr = _to_numpy_array(X)
    y_arr = _to_numpy_array(y)

    if len(X_arr) != len(y_arr):
        raise ValueError('X and y must have the same length')
    if num_clients < 1:
        raise ValueError('num_clients must be at least 1')

    rng = np.random.RandomState(seed)
    counts = Counter(y_arr.tolist())
    classes = sorted(counts.keys())
    minority_label = min(classes, key=lambda cls: (counts[cls], cls))

    minority_indices = list(np.flatnonzero(y_arr == minority_label))
    majority_indices = list(np.flatnonzero(y_arr != minority_label))
    rng.shuffle(minority_indices)
    rng.shuffle(majority_indices)

    client_ids = [f'client_{i:02d}' for i in range(num_clients)]
    num_zero_minority = max(1, num_clients // 5)
    num_few_minority = max(2, num_clients // 2)
    num_many_minority = num_clients - num_zero_minority - num_few_minority

    minority_requirements: List[int] = []
    for i in range(num_clients):
        if i < num_zero_minority:
            minority_requirements.append(0)
        elif i < num_zero_minority + num_few_minority:
            minority_requirements.append(rng.randint(1, 4))
        else:
            minority_requirements.append(rng.randint(4, max(4, int(len(minority_indices) / max(1, num_many_minority)))))

    rng.shuffle(minority_requirements)
    total_minority_needed = sum(minority_requirements)
    if total_minority_needed > len(minority_indices):
        scale = len(minority_indices) / total_minority_needed
        minority_requirements = [max(0, int(np.floor(count * scale))) for count in minority_requirements]
        for idx in range(len(minority_requirements)):
            if sum(minority_requirements) < len(minority_indices) and minority_requirements[idx] == 0:
                minority_requirements[idx] = 1
        minority_requirements = minority_requirements[:num_clients]

    majority_pool_size = len(majority_indices)
    majority_weights = rng.rand(num_clients)
    majority_weights /= majority_weights.sum()
    majority_requirements = [max(1, int(round(w * majority_pool_size))) for w in majority_weights]

    total_majority_assigned = sum(majority_requirements)
    if total_majority_assigned != majority_pool_size:
        diff = majority_pool_size - total_majority_assigned
        for i in range(abs(diff)):
            idx = i % num_clients
            majority_requirements[idx] += int(np.sign(diff))
            if majority_requirements[idx] < 1:
                majority_requirements[idx] = 1

    clients: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    remaining_minority = minority_indices.copy()
    remaining_majority = majority_indices.copy()

    for client_id, minority_count, majority_count in zip(client_ids, minority_requirements, majority_requirements):
        assigned_minority = []
        if minority_count > 0 and remaining_minority:
            take = min(minority_count, len(remaining_minority))
            assigned_minority = remaining_minority[:take]
            remaining_minority = remaining_minority[take:]

        take_majority = min(majority_count, len(remaining_majority))
        assigned_majority = remaining_majority[:take_majority]
        remaining_majority = remaining_majority[take_majority:]

        assigned_indices = assigned_minority + assigned_majority
        if not assigned_indices and remaining_majority:
            assigned_indices = [remaining_majority.pop(0)]

        client_X = X_arr[assigned_indices]
        client_y = y_arr[assigned_indices]
        clients[client_id] = (client_X, client_y)

    if remaining_minority or remaining_majority:
        leftovers = remaining_minority + remaining_majority
        for i, idx in enumerate(leftovers):
            client_id = client_ids[i % num_clients]
            client_X, client_y = clients[client_id]
            clients[client_id] = (
                np.vstack([client_X, X_arr[[idx]]]),
                np.concatenate([client_y, y_arr[[idx]]]),
            )

    print('Global training distribution:')
    _print_distribution('Training', y_arr)
    print('Per-client distribution:')
    for client_id, (_, client_y) in clients.items():
        _print_distribution(f'  {client_id}', client_y)

    return clients


def select_clients(
    clients: Dict[str, Tuple[np.ndarray, np.ndarray]],
    participation_rate: Union[float, int],
    seed: int = 42,
) -> List[str]:
    """Select a subset of clients for a training round."""
    client_ids = list(clients.keys())
    num_clients = len(client_ids)
    rng = np.random.RandomState(seed)

    if isinstance(participation_rate, float):
        if not 0 < participation_rate <= 1:
            raise ValueError('participation_rate must be a float between 0 and 1')
        num_selected = max(1, int(round(num_clients * participation_rate)))
    elif isinstance(participation_rate, int):
        num_selected = min(max(1, participation_rate), num_clients)
    else:
        raise ValueError('participation_rate must be a float or integer')

    selected = rng.choice(client_ids, size=num_selected, replace=False).tolist()
    print(f'Selected {len(selected)} / {num_clients} clients: {selected}')
    return selected


def prepare_federated_data(
    X: Any,
    y: Any,
    ratio: Union[str, Tuple[int, int]],
    num_clients: int,
    participation_rate: Union[float, int] = 1.0,
    seed: int = 42,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """Create federated data partitions and a held-out stratified test set."""
    X_imbalanced, y_imbalanced = create_imbalance(X, y, ratio, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X_imbalanced,
        y_imbalanced,
        test_size=0.3,
        seed=seed,
    )
    clients = split_non_iid_clients(X_train, y_train, num_clients=num_clients, seed=seed)
    if participation_rate is not None:
        select_clients(clients, participation_rate, seed=seed)
    print(f'Prepared {len(clients)} clients and test set of size {len(X_test)}')
    return clients, (X_test, y_test)
