from datetime import datetime
import sys
from sys import platform
from itertools import combinations
import subprocess
import platform
import os
from scipy.spatial import distance
import pickle
import re
import pandas as pd
import torch
import numpy as np
from typing import List, Union, Dict, Optional, Any
from pandera.typing import Series, DataFrame
from numpy.typing import NDArray
from log_config import base_logger

logger = base_logger(__name__)


def timer(start_time=None) -> str:
    """
    Return the time consumed between the start time and current time.

    :param start_time: Start time of the process
    :type start_time: datetime, optional
    :return: Completion time string
    :rtype: str
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        return f"Time consumption: {thour:.0f} hours {tmin:.0f} minutes and {tsec:.2f} seconds"


def get_base_prefix_compat():
    """
    Get the base/real prefix if it exists, otherwise return sys.prefix.

    :return: The base/real prefix if it exists, otherwise sys.prefix.
    """
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix


def in_virtualenv():
    """Check if the code is currently running inside a virtual environment.

    :return: (bool) True if inside a virtual environment, False otherwise.
    """
    return get_base_prefix_compat() != sys.prefix


def inspect_environment(REQUIRED_PYTHON: str = "python3"):
    """
    Check Python version for development environment.
    :param REQUIRED_PYTHON: (str) The required Python version to test.
    """
    print("**" * 40)
    logger.info("\nSYSTEM INFO:\n")
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized Python interpreter: {}".format(REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        logger.info(">>> Development environment passes all tests!")

    logger.info("Python Version: %s", platform.python_version())
    if in_virtualenv():
        logger.info(">>> Python Version is running on Virtual Environment!")
    else:
        logger.info("Python Version is not running on Virtual Environment!")

    print("**" * 40)


def validate_unique_features(features_list: List[str]) -> bool:
    """Check if any feature set is repeated in the list.

    :param features_list: List of features.
    :return: True if all feature sets are unique, False otherwise.
    """
    list_combinations = list()
    for n in range(len(features_list) + 1):
        list_combinations += list(combinations(features_list, n))
    for value in list_combinations:
        if len(value) == 2:
            if value[0] == value[1]:
                print("Two features are equal: {0}, {1}".format(value[0], value[1]))
                return False
    print("All feature sets are unique.")
    return True


def get_pip_package_version(package_name):
    """Returns the version of a package installed by pip.

    For example, if the package name 'numpy' is provided, this function will return the installed version of numpy.

    :param package_name: (str) The name of the package to check the version for.
    :return: (str) The version of the specified package.
    """
    command = f"pip freeze | grep -w {package_name}= | awk -F '==' '{{print $2}}' | tr -d '\n'"
    output = subprocess.check_output(command, shell=True).decode()
    return output.strip()


def install_pip_package(package_name):
    """
    Installs a pip package with a specified version if the package is not already installed.

    :param package_name: The name and version of the package to install.
    """
    package = package_name.split("==")[0]
    print(f"Previous version is {get_pip_package_version(package)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError as e:
        print('Failed to install the package')
    print(f"Current version {get_pip_package_version(package)}")


# Define a function to get the project root directory
def find_project_path() -> str:
    """
    Returns the absolute path of the project root directory.

    The method traverses up the directory hierarchy until it finds the file or folder that is unique to your project.
    In this case, it searches for the existence of a README.md file.

    :return: (str) Absolute path of the project root directory
    """
    # Get the current working directory
    current_dir = os.getcwd()

    # Traverse up the directory hierarchy until you find the file or folder that is unique to your project
    while not os.path.exists(os.path.join(current_dir, 'README.md')):
        current_dir = os.path.dirname(current_dir)

    return current_dir


def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two vectors a and b.
    :param a: Vector a (NumPy array or torch.Tensor)
    :param b: Vector b (NumPy array or torch.Tensor)
    :return: float: The Euclidean distance between vectors a and b.
    """
    is_torch = isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
    if is_torch:
        a = a.detach().numpy() if isinstance(a, torch.Tensor) else a
        b = b.detach().numpy() if isinstance(b, torch.Tensor) else b
    return distance.euclidean(a, b)


def get_available_gpus():
    # Check if GPUs are available by inspecting the "CUDA_VISIBLE_DEVICES" environment variable
    return len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


def check_array_torch_numpy(datapoint, model):
    if isinstance(datapoint, np.ndarray):
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                return model(torch.tensor(datapoint).float().reshape(1, -1)).numpy()
        else:
            return model.predict(datapoint.reshape(1, -1))
    elif isinstance(datapoint, torch.Tensor):
        with torch.no_grad():
            return model.predict_proba(datapoint.reshape(1, -1)).numpy()


def to_tensor(data: Union[DataFrame, Series, NDArray]) -> torch.Tensor:
    """
    Convert a Pandas DataFrame, Series, or NumPy array to a PyTorch tensor, or keep it as a tensor if already in PyTorch format.

    :param data: Input data, either a Pandas DataFrame, Series, NumPy array, or a PyTorch tensor.
    :return: PyTorch tensor.
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values

    if isinstance(data, np.ndarray):
        tensor_data = torch.tensor(data, dtype=torch.float32)
        return tensor_data
    elif isinstance(data, torch.Tensor):
        return torch.tensor(data, dtype=torch.float32)



    else:
        raise ValueError(
            "Unsupported data type. Input must be a Pandas DataFrame, "
            "Series, NumPy array, or a PyTorch tensor.")


def to_tensor_labels(data: Union[DataFrame, Series, NDArray, torch.Tensor], is_labels: bool = False) -> torch.Tensor:
    """
    Convert a Pandas DataFrame, Series, NumPy array, or torch tensor to a PyTorch tensor.
    If is_labels is True, the tensor will be casted to long type.

    :param data: Input data, either a Pandas DataFrame, Series, NumPy array, or a PyTorch tensor.
    :param is_labels: Whether the data represents labels (default is False).
    :return: PyTorch tensor.
    """
    tensor_data = to_tensor(data)

    if is_labels:
        tensor_data = tensor_data.long()

    return tensor_data


def to_numpy(data: Union[torch.tensor, pd.Series, pd.DataFrame, NDArray]) -> NDArray:
    """
    Convert a Pandas Series, DataFrame, or PyTorch tensor to a NumPy array.

    :param data: Input data, either a Pandas Series, DataFrame, or a PyTorch tensor.
    :return: NumPy array.
    """
    if isinstance(data, pd.Series):  # Check if it's a Pandas Series
        data = data.to_numpy()  # Use the to_numpy() method
    elif isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass  # Input data is already a NumPy array. No conversion needed.
    else:
        raise ValueError("Unsupported data type. Input must be a Pandas Series, DataFrame, or a PyTorch tensor.")
    return data


def count_feature_categories(input_dict: Dict[str, List[str]]) -> Dict[str, int]:
    """

    :return:
    :param input_dict: Input dictionary where keys are the names of the datapoints and values are lists of features.
    :return: dictionary with the count of each feature category.
    """
    categories_count = {
        'HpHp': 0,
        'HH': 0,
        'MI': 0,
        'H': 0,
        'HH_jit': 0,
    }

    for feature_list in input_dict.values():
        for feature in feature_list:
            if 'HpHp' in feature:
                categories_count['HpHp'] += 1
            if 'HH' in feature:
                categories_count['HH'] += 1
            if 'MI' in feature:
                categories_count['MI'] += 1
            if 'H_' in feature:  # This ensures it matches 'H_L0.1_weight', 'H_L1_weight', etc.
                categories_count['H'] += 1
            if 'HH_jit' in feature:
                categories_count['HH_jit'] += 1

    return categories_count


def network_categories_features(list_features: List[str]) -> Dict[str, List[str]]:
    """

    :param list_features: List of features
    :return: Dictionary with the features of each network category.
    """
    network_categories = {
        'HpHp': [],
        'HH': [],
        'MI': [],
        'H': [],
        'HH_jit': [],
    }
    for feature in list_features:
        if feature.startswith('MI'):
            network_categories['MI'].append(feature)
        if feature.startswith('HH'):
            if feature.startswith('HH_jit'):
                network_categories['HH_jit'].append(feature)
            else:
                network_categories['HH'].append(feature)
        if feature.startswith('HpHp'):
            network_categories['HpHp'].append(feature)
        if feature.startswith('H_'):
            network_categories['H'].append(feature)
    return network_categories


def load_scikit_model(model_file_path: str):
    """
    Load a scikit-learn model from a file.
    :param model_file_path: Path to the model file.
    :return: The loaded model object.
    """
    try:
        with open(model_file_path, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        return loaded_model
    except Exception as e:
        print(f"Error loading the model from {model_file_path}: {e}")
        return None


def saving_file_path(directory_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - directory_path (str): The path of the directory to be created.
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")


def create_directory_path(folder_path: str):
    """
    Create a folder if it does not exist and return the folder path.
    Parameters:
    folder_path (str): The path of the folder to create.
    Returns:
    str: The path of the created folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def get_device():
    """
    Get the device (CPU or GPU) that PyTorch will use for computation.
    :return: PyTorch device object.
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device) -> Union[
    torch.Tensor, List[torch.Tensor]]:
    """
    Move data to the specified device.

    :param data: Data to be moved to the device.
    :param device: Device to move the data to.
    :return: Data moved to the device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def convert_medbiot_to_nbiot_features(features: List[str]) -> Dict[str, str]:
    """
    :param features: List of features
    :return: dict with the features converted to the nbiot format.
    """
    features_dict = {}
    # trim_features = [ for s in features]
    for feature in features:
        feature_split = feature.split('_')
        if feature.startswith('MI'):
            feature_split[2] = 'L' + feature_split[2]
        if feature.startswith('HH'):
            if feature.startswith('HH_jit'):
                feature_split[2] = 'L' + feature_split[2]
            else:
                feature_split[1] = 'L' + feature_split[1]
        if feature.startswith('HpHp'):
            feature_split[1] = 'L' + feature_split[1]
        new_feature = '_'.join(feature_split)
        final_feature = re.sub(r'_\d+(_\d+)*$', '', new_feature)
        features_dict[feature] = final_feature
    return features_dict