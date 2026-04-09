from collections import OrderedDict
import torch
from src.fedLearn.centralized import fed_test

def get_on_fit_config(client_configs):
    def fit_config_fn(server_round: int):
        return {
            "hidden_layers": client_configs['hidden_layers'],
            "hidden_units": client_configs['hidden_units'],
            "learning_rate": client_configs['learning_rate'],
            "optimizer": client_configs['optimizer'],
            "activation": client_configs['activation'],
            "batch_size": client_configs['batch_size'],
            "epochs": client_configs['epochs'],
            "round_id": server_round
        }
    return fit_config_fn

def get_evaluate_server_fn(model, test_loader):
    def evaluate_fn(server_round, parameters, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v, device=device) for k, v in params_dict
        })
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = fed_test(model=model, test_loader=test_loader)
        print(f"Server round: {server_round}, Loss: {loss}, Accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}
    return evaluate_fn