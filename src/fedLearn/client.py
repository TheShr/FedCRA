import torch.optim as optim
from collections import OrderedDict
from flwr.common import NDArrays, Scalar
import torch
from src.fedLearn.centralized import fed_train, fed_test
from log_config import base_logger
import xgboost as xgb
import flwr as fl
import os
import json
import time
import numpy as np
from pathlib import Path
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")

logger = base_logger(__name__)


class ClientModel(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, test_loader, client_names, results_path=None):
        super().__init__()
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.client_names = client_names
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # Automatically select GPU if available
        self.results_path = results_path
        self.round_id = 0

        if 0 <= int(client_id) < len(client_names):
            self.client_name = client_names[int(client_id)]
        else:
            self.client_name = "client"

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.device) for k, v in params_dict})  # Move parameters to GPU if available
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: dict):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in
                self.model.state_dict().items()]  # Ensure parameters are on CPU for transmission

    def fit(self, parameters, config):
        self.set_parameters(parameters)  # Set model parameters received from the server

        lr = config['learning_rate']
        self.round_id = config.get('round_id')  # Always set 'round_id' from conf
        optimizer_conf = config['optimizer']
        optimizer = getattr(optim, optimizer_conf)(self.model.parameters(), lr=lr)
        epochs = config['epochs']

        print(f"Client {self.client_name} training for round {self.round_id}")
        start_time = time.time()

        # Move model to the device (GPU/CPU)
        self.model.to(self.device)

        # Train the model using the federated training function
        metrics = fed_train(model=self.model, epochs=epochs, optimizer=optimizer, train_loader=self.train_loader)
        end_time = time.time()
        communication_time = end_time - start_time

        # Log results and save model state
        self.write_results_json(metrics, communication_time, phase="train", round_id=self.round_id)
        self.save_model(self.model, round_id=self.round_id)

        return self.get_parameters({}), len(self.train_loader), {}

    def evaluate(self, parameters: NDArrays, config: dict):
        self.set_parameters(parameters)  # Set model parameters received from the server
        print(f"Client {self.client_name} evaluating for round {self.round_id}")

        start_time = time.time()

        # Move model to the device (GPU/CPU)
        self.model.to(self.device)

        # Evaluate the model using the federated evaluation function
        metrics = fed_test(self.model, self.test_loader)
        end_time = time.time()
        communication_time = end_time - start_time

        # Log results
        self.write_results_json(metrics, communication_time, phase="test", round_id=self.round_id)
        return float(metrics["loss"]), len(self.test_loader), {"accuracy": metrics["accuracy"]}

    def write_results_json(self, metrics, communication_time, phase, round_id):
        """Write the results of training/evaluation to a JSON file."""
        file_name = f"{self.results_path}/metrics/{self.client_name}.json"
        results = {
            "round": round_id,
            f"{phase}_metrics": metrics,
            "communication_time": communication_time
        }

        if os.path.exists(file_name):
            with open(file_name, "r") as json_file:
                data = json.load(json_file)
        else:
            data = []

        data.append(results)
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def save_model(self, model, round_id):
        """Save the model state to the local storage."""
        if not os.path.exists(f"{self.results_path}/clients"):
            os.makedirs(f"{self.results_path}/clients")
        torch.save(model.state_dict(), f"{self.results_path}/clients/{self.client_name}_rnd_{round_id}.pth")


def generate_client_fn(model, train_loaders, test_loaders, client_names, results_path):
    """Generate a function to create client instances for federated learning."""

    def client_fn(client_id):
        try:
            return ClientModel(client_id=client_id, model=model,
                               train_loader=train_loaders[int(client_id)],
                               test_loader=test_loaders[int(client_id)],
                               client_names=client_names,
                               results_path=results_path).to_client()
        except Exception as e:
            logger.error(f"Error occurred in client {client_id}: {e}")
            raise

    return client_fn


class XgbFedClient(fl.client.Client):
    """
    XGBoost client implementation for FedLearn
    """
    def __init__(self, client_name: str,
                 train_matrix: xgb.DMatrix,
                 val_matrix: xgb.DMatrix,
                 num_train: int, num_val: int,
                 num_local_round: int,
                 params: dict,
                 model_results_path: str = None):
        self.bst = None
        self.config = None
        self.client_name = client_name
        self.train_matrix = train_matrix
        self.val_matrix = val_matrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.current_round = 0
        self.params = params
        self.model_results_path = model_results_path
        self.results = []

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        for _ in range(self.num_local_round):
            self.bst.update(self.train_matrix, self.bst.num_boosted_rounds())
        bst = self.bst[self.bst.num_boosted_rounds() - self.num_local_round: self.bst.num_boosted_rounds()]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        self.current_round += 1
        if not self.bst:
            bst = xgb.train(
                self.params,
                self.train_matrix,
                num_boost_round=self.num_local_round,
                evals=[(self.val_matrix, "validate"), (self.train_matrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)
            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        results_path = Path(self.model_results_path) / 'clients'
        results_path.mkdir(parents=True, exist_ok=True)
        model_path = results_path / f'{self.client_name}_rnd_{self.current_round}.json'
        self.bst.save_model(model_path)
        print(f"Model saved to {model_path}")

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.bst is None:
            raise ValueError("Model has not been trained. 'self.bst' is None.")

        eval_results = self.bst.eval_set(evals=[(self.val_matrix, "valid")], iteration=self.bst.num_boosted_rounds() - 1)
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        y_true = self.val_matrix.get_label()
        y_pred_proba = self.bst.predict(self.val_matrix)
        print(f"Prediction shape: {y_pred_proba.shape}")
        if len(y_pred_proba.shape) == 1:
            y_pred = [1 if pred > 0.5 else 0 for pred in y_pred_proba]
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        metrics = {
            "round": self.current_round,
            "AUC": auc,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }

        try:
            results_path = Path(self.model_results_path) / 'metrics'
            print(f"Creating directory: {results_path}")
            results_path.mkdir(parents=True, exist_ok=True)
            metrics_path = results_path / f'{self.client_name}.json'
            print(f"Saving metrics to: {metrics_path}")
            self.results.append(metrics)
            with open(metrics_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0,
            num_examples=self.num_val,
            metrics=metrics,
        )