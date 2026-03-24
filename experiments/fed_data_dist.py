from src.fedLearn.fed_data import split_clients
from log_config import base_logger
import pandas as pd
import numpy as np

logger = base_logger(__name__)

data_folder = 'data/cic_iomt'
data_file = 'cic_iomt.csv.bz2'
label_name = 'Category'
num_clients = 10

#%%
client_splits = split_clients(data_folder, data_file, label_name, num_clients)
{print(f"Client {i}: {c_df.shape}, {c_df['Binary'].value_counts()}") for i, c_df in enumerate(client_splits)}

#%%
# import path
from pathlib import Path




def split_clients_non_iid(data_folder: str, data_file: str, label_name: str, num_clients: int = 10,
                          imbalance_factor: float = 0.5):
    """
    Split the dataset into Non-IID client datasets for federated learning.
    Clients will have imbalanced distributions of the labels.

    :param data_folder: Path to the data folder.
    :param data_file: File name of the dataset.
    :param label_name: The name of the label column.
    :param num_clients: Number of clients.
    :param imbalance_factor: Determines how imbalanced the class distribution is for each client.
    :return: List of data splits for each client.
    """
    logger.info(f"Data File Location: {data_folder}")
    logger.info(f"Data File NAME: {data_file}")
    try:
        df = pd.read_csv(Path().joinpath(data_folder, data_file))
    except FileNotFoundError:
        error_msg = f"File '{data_file}' not found in DataLocation '{data_folder}'."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Group by label
    grouped = df.groupby(label_name)

    # Create splits
    splits = [pd.DataFrame(columns=df.columns) for _ in range(num_clients)]

    # Non-IID: Distribute labels unevenly among clients
    for label, group in grouped:
        # Split into unequal proportions
        proportions = np.random.dirichlet([imbalance_factor] * num_clients)
        split_groups = [group.iloc[:int(len(group) * p)] for p in proportions]

        # Assign data to clients
        for i, split_group in enumerate(split_groups):
            splits[i] = pd.concat([splits[i], split_group])
            group = group.iloc[len(split_group):]  # Remove assigned rows

    # Shuffle each client dataset to mix the assigned data
    for i in range(num_clients):
        splits[i] = splits[i].sample(frac=1, random_state=42).reset_index(drop=True)

    # total all the samples count how many smaple combined in all the clients
    total_samples = sum([len(client_df) for client_df in splits])
    print(f"Total samples: {total_samples}")

    return splits

#%%
# ==== 0) Imports & config ====
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np, math
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.deepLearn.model import  LSTMModel
from src.fedLearn.centralized import fed_train, fed_test


from src.dataLoaders.data_loaders import  load_sample_data

data, targets = load_sample_data(
    folder_name=data_folder,
    file_name=data_file,
    class_name=label_name,
    sample_size=5000,
    n_features=45
)
#%%
# split the data into train and test sets
train_data, test_data, train_targets, test_targets = train_test_split(
    data, targets, test_size=0.2, random_state=42
)

# Scale the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Convert to PyTorch tensors
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
train_targets_tensor = torch.tensor(train_targets.to_numpy(), dtype=torch.long)
test_targets_tensor = torch.tensor(test_targets.to_numpy(), dtype=torch.long)


# Create DataLoader for training and testing
train_dataset = TensorDataset(train_data_tensor, train_targets_tensor)
test_dataset = TensorDataset(test_data_tensor, test_targets_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = train_data_tensor.shape[2]  # Number of features
output_size = len(np.unique(train_targets))  # Number of classes
hidden_units = 200  # Number of hidden units in LSTM
model = LSTMModel(input_size, output_size, hidden_units, hidden_layers=5)

optimzer = torch.optim.Adam(model.parameters(), lr=0.001)

train_metrics = fed_train(
    model=model,
    epochs=500,
    optimizer=optimzer,
    train_loader=train_loader,
)

test_metrics = fed_test(
    model=model,
    test_loader=test_loader
)