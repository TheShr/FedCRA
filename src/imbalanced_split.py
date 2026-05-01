import random
import numpy as np
from torch.utils.data import Dataset, Subset

def create_two_client_imbalance(dataset, ratio=(1, 10), shuffle=True, seed=42):
    """
    Splits dataset into 2 clients with a specified imbalance ratio.

    Args:
        dataset: Full dataset (list, numpy array, or PyTorch Dataset)
        ratio: Tuple (r1, r2) representing data proportion between client 1 and client 2
        shuffle: Whether to shuffle dataset before splitting
        seed: Random seed for reproducibility

    Returns:
        client_1_data, client_2_data: Two disjoint subsets of the dataset
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Normalize ratio
    r1, r2 = ratio
    total = r1 + r2

    # Compute split sizes
    dataset_size = len(dataset)
    size1 = max(1, int(dataset_size * (r1 / total)))  # Ensure at least 1 sample
    size2 = dataset_size - size1

    # Create indices
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)

    indices1 = indices[:size1]
    indices2 = indices[size1:]

    # Split based on dataset type
    if isinstance(dataset, Dataset):
        # PyTorch Dataset
        client_1_data = Subset(dataset, indices1)
        client_2_data = Subset(dataset, indices2)
    elif isinstance(dataset, np.ndarray):
        # NumPy array
        client_1_data = dataset[indices1]
        client_2_data = dataset[indices2]
    elif isinstance(dataset, list):
        # Python list
        client_1_data = [dataset[i] for i in indices1]
        client_2_data = [dataset[i] for i in indices2]
    else:
        raise ValueError("Unsupported dataset type. Must be PyTorch Dataset, NumPy array, or Python list.")

    # Logging
    print(f"Client 1 size: {len(client_1_data)}, Client 2 size: {len(client_2_data)}")
    print(f"Imbalance ratio: {ratio} (normalized: {r1/total:.4f}, {r2/total:.4f})")

    return client_1_data, client_2_data

def create_multiple_imbalanced_splits(dataset, ratios=[(1,1), (1,10), (1,100), (1,1000)], shuffle=True, seed=42):
    """
    Creates multiple imbalanced splits for different ratios.

    Args:
        dataset: Full dataset
        ratios: List of tuples (r1, r2)
        shuffle: Whether to shuffle dataset before splitting
        seed: Random seed for reproducibility

    Returns:
        List of (client_1_data, client_2_data) for each ratio
    """
    splits = []
    for ratio in ratios:
        client1, client2 = create_two_client_imbalance(dataset, ratio=ratio, shuffle=shuffle, seed=seed)
        splits.append((client1, client2))
    return splits