from log_config import base_logger
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import requests
from io import BytesIO
import numpy as np
from urllib.parse import urlparse
from typing import Any, Union, Tuple

logger = base_logger(__name__)


def choose_num_workers(loaders_per_process: int = 2, hard_cap: int = 8) -> int:
    """Auto-detect optimal number of workers based on available CPUs."""
    try:
        cpus_alloc = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    except ValueError:
        cpus_alloc = 0
    cpus = cpus_alloc if cpus_alloc > 0 else (os.cpu_count() or hard_cap)
    cap = min(hard_cap, cpus)
    return max(0, cap // max(1, loaders_per_process))


_NUM_WORKERS = choose_num_workers(loaders_per_process=2)


def load_sample_data(folder_name: str, file_name: str, class_name: str, sample_size: int, n_features: int) -> Tuple[
    pd.DataFrame, pd.Series]:
    """
    Read dataframe from a folder and sample it.
    :param folder_name: The folder containing the dataframe.
    :param file_name: The name of the CSV file containing the dataframe.
    :param class_name: The name of the column containing the class labels.
    :param sample_size: The number of samples to take from each class.
    :param n_features: The number of features to use.
    :return: DataFrame, Data, and Target Values pd.Series
    """
    logger.info(f"Data File Location: {folder_name}")
    logger.info(f"Data File NAME: {file_name}")

    try:
        dataframe = pd.read_csv(Path().joinpath(folder_name, file_name))
    except FileNotFoundError:
        error_msg = f"File '{file_name}' not found in DataLocation '{folder_name}'."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    label_column = dataframe[class_name]
    if len(label_column.unique()) == 2:
        logger.info("Classification Type: Binary")
    else:
        logger.info("Classification Type: Multi-Class")

    logger.info(f"DATA SIZE: {dataframe.shape}")
    logger.info(f"Class Name: {class_name}")

    # Sample data
    df = dataframe.groupby(class_name).apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)

    # Encode class labels
    le = LabelEncoder()
    df[class_name] = le.fit_transform(df[class_name])
    logger.info(f"Class Labels Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    # Print the label mapping clearly
    logger.info("Class Labels Mapping:")
    for original_class, encoded_class in zip(le.classes_, le.transform(le.classes_)):
        logger.info(f"{original_class} : {encoded_class}")
    # Split data and target
    data = df.iloc[:, :n_features]
    target_values = df[class_name]
    return data, target_values



class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        super(LoadDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        return data, targets

    def __len__(self):
        return len(self.targets)


def get_torch_loader(data: torch.Tensor, labels: torch.Tensor, batch_size: int = 1024,
                     shuffle: bool = True) -> DataLoader:
    """
    Get a torch DataLoader optimized for GPU with efficient data transfer.
    
    Args:
        data: Torch tensor of input data.
        labels: Torch tensor of target values.
        batch_size: int, batch size.
        shuffle: bool, whether to shuffle the data.

    Returns: DataLoader with GPU optimizations:
        - pin_memory: Enables fast CPU->GPU transfer
        - num_workers: Parallel data loading 
        - persistent_workers: Keeps worker processes alive between batches
        - prefetch_factor: Prefetch multiple batches

    """
    tensor_data = TensorDataset(data.cpu(), labels.cpu())
    return DataLoader(
        tensor_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=4 if _NUM_WORKERS > 0 else None,
    )


def loader_to_data_labels(data_loader: DataLoader):
    extracted_data = []
    extracted_labels = []
    for batch_data, batch_labels in data_loader:
        extracted_data.append(batch_data)
        extracted_labels.append(batch_labels)
    extracted_data = torch.cat(extracted_data, dim=0)
    extracted_labels = torch.cat(extracted_labels, dim=0)
    return extracted_data, extracted_labels


# def get_torch_dataloaders(data: NDArray, target: NDArray, batch_size: int) -> Tuple[DataLoader, DataLoader]:
#     """
#     Get Torch DataLoaders for training and testing.
#     Data is already split into train and test sets.
#     :param data: Input data.
#     :param target: Target values.
#     :param batch_size: Batch size.
#     :return: Torch DataLoaders for training and testing.
#     """
#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
#     pre_processing = DataPreprocessing()
#     logger.debug("Data Normalization .....! done")
#     normalized_X_train = pre_processing.min_max_normalization(X_train)
#     normalized_X_test = pre_processing.min_max_normalization(X_test)
#
#     y_train_np = to_tensor(y_train).to(torch.long)
#     y_test_np = to_tensor(y_test).to(torch.long)
#     normalized_X_train = to_tensor(normalized_X_train).to(torch.float32)
#     normalized_X_test = to_tensor(normalized_X_test).to(torch.float32)
#
#     train_dataset = LoadDataset(normalized_X_train, y_train_np)
#     test_dataset = LoadDataset(normalized_X_test, y_test_np)
#
#     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_data_loader, test_data_loader
#

def load_data_from_url(url: str, save_path: str = None) -> Any:
    """
    Load data from a URL.
    :param url:  URL to load data from
    :param save_path:  Path to save the data to
    :return: data from url (pd.DataFrame, np.ndarray, torch.Tensor)
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)  # Extract file name from the URL
        _, file_extension = os.path.splitext(file_name)
        if parsed_url.netloc == 'github.com':
            # If the URL is from GitHub, modify the URL to get the raw content
            raw_url = f'https://raw.githubusercontent.com{parsed_url.path.replace("/blob/", "/")}'
            response = requests.get(raw_url)
            response.raise_for_status()

        if file_extension.lower() == '.csv':
            file_content = BytesIO(response.content)
            data = pd.read_csv(file_content)
        elif file_extension.lower() == '.xls':
            file_content = BytesIO(response.content)
            data = pd.read_excel(file_content)
        elif file_extension.lower() == '.pt':
            file_content = BytesIO(response.content)
            data = torch.load(file_content, map_location=torch.device('cpu'))  # Specify device if needed
        elif file_extension.lower() == '.npy':
            file_content = BytesIO(response.content)
            data = np.load(file_content)
        else:
            raise ValueError(f'Unsupported file extension: {file_extension}')
        if save_path is not None:
            # Construct the full save path
            full_save_path = os.path.join(save_path, file_name)
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            with open(full_save_path, 'wb') as f:
                f.write(response.content)
        return data
    except Exception as e:
        print(f'Error loading data from {url}: {e}')
        return None



class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        super(LoadDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        return data, targets

    def __len__(self):
        return len(self.targets)


def loader_to_data_labels(data_loader: DataLoader):
    extracted_data = []
    extracted_labels = []
    for batch_data, batch_labels in data_loader:
        extracted_data.append(batch_data)
        extracted_labels.append(batch_labels)
    extracted_data = torch.cat(extracted_data, dim=0)
    extracted_labels = torch.cat(extracted_labels, dim=0)
    return extracted_data, extracted_labels


# def get_torch_dataloaders(data: NDArray, target: NDArray, batch_size: int) -> Tuple[DataLoader, DataLoader]:
#     """
#     Get Torch DataLoaders for training and testing.
#     Data is already split into train and test sets.
#     :param data: Input data.
#     :param target: Target values.
#     :param batch_size: Batch size.
#     :return: Torch DataLoaders for training and testing.
#     """
#     # Split the data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
#     pre_processing = DataPreprocessing()
#     logger.debug("Data Normalization .....! done")
#     normalized_X_train = pre_processing.min_max_normalization(X_train)
#     normalized_X_test = pre_processing.min_max_normalization(X_test)
#
#     y_train_np = to_tensor(y_train).to(torch.long)
#     y_test_np = to_tensor(y_test).to(torch.long)
#     normalized_X_train = to_tensor(normalized_X_train).to(torch.float32)
#     normalized_X_test = to_tensor(normalized_X_test).to(torch.float32)
#
#     train_dataset = LoadDataset(normalized_X_train, y_train_np)
#     test_dataset = LoadDataset(normalized_X_test, y_test_np)
#
#     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_data_loader, test_data_loader
#

def load_data_from_url(url: str, save_path: str = None) -> Any:
    """
    Load data from a URL.
    :param url:  URL to load data from
    :param save_path:  Path to save the data to
    :return: data from url (pd.DataFrame, np.ndarray, torch.Tensor)
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)  # Extract file name from the URL
        _, file_extension = os.path.splitext(file_name)
        if parsed_url.netloc == 'github.com':
            # If the URL is from GitHub, modify the URL to get the raw content
            raw_url = f'https://raw.githubusercontent.com{parsed_url.path.replace("/blob/", "/")}'
            response = requests.get(raw_url)
            response.raise_for_status()
        if file_extension.lower() == '.csv':
            file_content = BytesIO(response.content)
            data = pd.read_csv(file_content)
        elif file_extension.lower() == '.xls':
            file_content = BytesIO(response.content)
            data = pd.read_excel(file_content)
        elif file_extension.lower() == '.pt':
            file_content = BytesIO(response.content)
            data = torch.load(file_content, map_location=torch.device('cpu'))  # Specify device if needed
        elif file_extension.lower() == '.npy':
            file_content = BytesIO(response.content)
            data = np.load(file_content)
        else:
            raise ValueError(f'Unsupported file extension: {file_extension}')
        if save_path is not None:
            full_save_path = os.path.join(save_path, file_name)
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            with open(full_save_path, 'wb') as f:
                f.write(response.content)
        return data
    except Exception as e:
        print(f'Error loading data from {url}: {e}')
        return None






