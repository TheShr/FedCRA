import torch
from collections import OrderedDict
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import Scalar, NDArray
from src.ganLearn.centralized import fed_train_gan, fed_test_gan
import os
import json
import time

class GANClientModel(fl.client.NumPyClient):
    def __init__(self, client_id, client_names, gan_model,  train_loader, val_loader, noise_dim=100,  results_path=None):
        super().__init__()
        self.gan_model = gan_model
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.noise_dim = noise_dim
        self.client_id = client_id
        self.client_names = client_names
        self.results_path = results_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 0 <= int(client_id) < len(client_names):
            self.client_name = client_names[int(client_id)]
        else:
            self.client_name = "client"

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.gan_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.gan_model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.gan_model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model with the provided parameters."""

        learning_rate = config["learning_rate"]
        self.round_id = config.get('round_id')
        print(f"Client {self.client_name} training for round {self.round_id}")

        self.gan_model.to(self.device)
        gen_optimizer = torch.optim.Adam(self.gan_model.generator.parameters(), lr=learning_rate)
        disc_optimizer = torch.optim.Adam(self.gan_model.discriminator.parameters(), lr=learning_rate)

        start_time = time.time()
        metrics  = fed_train_gan(
            gan_model=self.gan_model,
            train_loader = self.train_loader,
            epochs = config["epochs"],
            noise_dim = self.noise_dim,
            gen_optimizer=gen_optimizer,
            disc_optimizer= disc_optimizer
        )
        end_time = time.time()
        communication_time = end_time - start_time
        self.write_results(
            metrics=metrics,
            communication_time=communication_time,
            phase="train",
            round_id=self.round_id
        )
        return self.get_parameters({}), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model with the provided parameters."""
        self.set_parameters(parameters)
        start_time = time.time()
        loss, wasserstein_distance  = fed_test_gan(
            gan_model=self.gan_model,
            test_loader = self.test_loader,
            noise_dim=self.noise_dim
        )
        end_time = time.time()
        communication_time = end_time - start_time
        return float(loss), len(self.test_loader), {"w_distance": wasserstein_distance}


    def write_results(self, metrics, communication_time, phase, round_id):
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
            json.dump(data, json_file)

def generate_client_fn(client_names, gan_model, train_loaders, val_loaders, results_path ,  noise_dim=100):
    def client_fn(cid: str):
        train_loader = train_loaders[int(cid)]
        test_loader = val_loaders[int(cid)]
        return GANClientModel(
            client_id=cid,
            client_names=client_names,
            gan_model=gan_model,
            train_loader=train_loader,
            val_loader=test_loader,
            results_path=results_path,
            noise_dim=noise_dim
        ).to_client()
    return client_fn



