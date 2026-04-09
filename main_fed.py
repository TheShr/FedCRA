import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import flwr as fl
from src.fedLearn.server.server_side import get_on_fit_config
from src.fedLearn.clients.nn_client import generate_client_fn
from src.fedLearn.server.server_side import get_evaluate_server_fn
import warnings
import pickle
from pathlib import Path
import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import json
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.fedLearn.fed_data import federated_data
from log_config import base_logger
from flwr.common.parameter import ndarrays_to_parameters

import warnings, re
warnings.filterwarnings(
    "ignore",
    message=re.escape("This DataLoader will create") + ".*",
    category=UserWarning,
    module="torch.utils.data.dataloader"
)


logger = base_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Auto-detect GPU and use it if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
torch.set_default_tensor_type(torch.FloatTensor)


def model_to_parameters(model):
    """Note that the model is already instantiated when passing it here.

    This happens because we call this utility function when instantiating the parent
    object (i.e. the FedAdam strategy in this example).
    """
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters

def evaluate_server_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total_samples = 0
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    # compute false

    cm = confusion_matrix(all_labels, all_predictions)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    FPR = FP / (FP + TN + 1e-10)
    macro_fpr = float(np.mean(FPR))  # Mean FPR across all classes


    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_fpr": macro_fpr
    }


def save_server_metrics_json(metrics, server_round, results_path, communication_time):


    metrics_dir = Path(results_path) / "metrics"
    file_name = metrics_dir / "server_metrics.json"

    # Ensure the directory exists
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
    else:
        data = []

    data.append({
        "round": server_round,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "macro_fpr": metrics["macro_fpr"],
        "communication_time": communication_time
    })

    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)


def get_evaluate_server_fn(model, test_loader, results_path):
    def evaluate_fn(server_round, parameters, config):
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(device) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)

        metrics = evaluate_server_model(model, test_loader)
        end_time = time.time()
        communication_time = end_time - start_time
        save_server_metrics_json(metrics, server_round, results_path, communication_time)
        if device.type == "cuda":
            logger.info(f"Server eval round {server_round} - GPU Memory Used: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        return metrics["loss"], {"accuracy": metrics["accuracy"]}

    return evaluate_fn


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'  # Ensure full error reporting

    print(OmegaConf.to_yaml(cfg))

    client_train_loaders, client_val_loaders, serv_train_loader, ser_test_loader = federated_data(
        data_folder=cfg.data_config.folder_name,
        data_file=cfg.data_config.file_name,
        label_name=cfg.data_config.label_name,
        n_features=cfg.data_config.n_features,
        num_clients=cfg.fed_config.num_clients,
        train_batch_size=512
    )

    device = torch.device("cpu")  # Force CPU usage
    model = instantiate(cfg.model).to(device)
    save_path = cfg.fed_config.model_results_path
    # saving path
    print("Main root folder:", save_path)

    model_name = model.__class__.__name__
    server_model_dir = f"{save_path}/{model_name}/server"

    # Ensure directories exist
    os.makedirs(server_model_dir, exist_ok=True)

    logger.info(f"Server model directory created: {server_model_dir}")


    client_names = [f"c{i+1}" for i in range(cfg.fed_config.num_clients)]

    client_fn = generate_client_fn(
        model=model,
        train_loaders=client_train_loaders,
        test_loaders=client_val_loaders,
        client_names=client_names,
        results_path=f'{save_path}/{model_name}'
    )

    strategy_name = cfg.fed_config.fed_strategy
    # Map string to strategy class
    strategy_map = {
        "fedAdagrad": fl.server.strategy.FedAdagrad,
        "fedAdam": fl.server.strategy.FedAdam,
        "fedAvg": fl.server.strategy.FedAvg,
        "fedAvgM": fl.server.strategy.FedAvgM,
        "fedMedian": fl.server.strategy.FedMedian,
        "fedOpt": fl.server.strategy.FedOpt,
        "fedYogi": fl.server.strategy.FedYogi,
        "fedKrum": fl.server.strategy.Krum,
        "fedTrimmedMean": fl.server.strategy.FedTrimmedAvg

    }

    base_strategy = strategy_map.get(strategy_name, fl.server.strategy.FedAvg)



    class SaveModelStrategy(base_strategy):
        def aggregate_fit(self, server_round: int, results, failures):
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                logger.info(f"Saving round {server_round} aggregated parameters...")

                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict(
                    {k: torch.tensor(v) for k, v in params_dict}
                )
                model.load_state_dict(state_dict, strict=True)
                model_save_path = f"{server_model_dir}/sm_{server_round}.pth"
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Server model saved at: {model_save_path}")

            return aggregated_parameters, aggregated_metrics

    # strategy name
    metrics_path = f"{save_path}/{model_name}"
    strategy = SaveModelStrategy(
        fraction_fit=0.1,
        min_fit_clients=cfg.fed_config.num_clients_per_round_fit,
        fraction_evaluate=0.1,
        min_evaluate_clients=cfg.fed_config.num_clients_per_round_eval,
        min_available_clients=cfg.fed_config.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_server_fn(model=model, test_loader=ser_test_loader, results_path=metrics_path),
        initial_parameters=model_to_parameters(model)
    )

    client_resources = {
        "num_cpus": cfg.fed_config.num_cpus,
        "num_gpus": 0,  # Force CPU usage
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.fed_config.num_clients,
        config=fl.server.ServerConfig(
            num_rounds=cfg.fed_config.num_rounds
        ),
        strategy=strategy,
        client_resources=client_resources,
    )

    # results_path = cfg.fed_config.model_results_path
    # history_file_path = Path(results_path) / f"{model_name}/{cfg.fed_config.fed_strategy}/history_fed_results_{cfg.dataset.dataset_name}.pkl"
    #
    # with open(history_file_path, "wb") as h:
    #     pickle.dump({"history": history}, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()