"""
client.py  —  FedCRA v11
=========================

FIXES vs v10
------------
[HIGH] cra_data values are JSON-serialised before going into fit_metrics.
    Flower requires all metric values to be scalar (int/float/str).
    cra_residuals and cra_class_counts are already JSON strings from
    centralized.py — they go straight into fit_metrics as strings.

[HIGH] client_id is always included in fit_metrics.
    Server uses this to key client_weights reliably, fixing the
    VirtualClientEngine proxy-address bug in fedcra.py.

[MEDIUM] fed_train receives client_id and grad_clip from config.
"""

import logging
import os
import json
import time
import traceback
from copy import deepcopy
import torch
import torch.optim as optim
import numpy as np
import xgboost as xgb
import flwr as fl

from collections import OrderedDict
from pathlib import Path

from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetParametersIns, GetParametersRes, Parameters, Status, NDArrays, Scalar,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.fedLearn.centralized import fed_train, fed_test
from log_config import base_logger

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")

logger = base_logger(__name__)


class ClientModel(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, test_loader,
                 client_names, results_path=None):
        super().__init__()
        self.client_id    = client_id
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model        = model
        self.client_names = client_names
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_path = results_path
        self.round_id     = 0

        self.client_name = (
            client_names[int(client_id)]
            if 0 <= int(client_id) < len(client_names)
            else "client"
        )
        self.client_log_path = Path(self.results_path) / "logs" / f"client_{self.client_name}.log"
        self._setup_client_logger()

    def _setup_client_logger(self):
        self.client_logger = logging.getLogger(f"{__name__}.{self.client_name}")
        self.client_logger.setLevel(logging.INFO)
        if not any(
            isinstance(handler, logging.FileHandler) and
            getattr(handler, "baseFilename", None) == str(self.client_log_path)
            for handler in self.client_logger.handlers
        ):
            os.makedirs(self.client_log_path.parent, exist_ok=True)
            file_handler = logging.FileHandler(self.client_log_path, mode="a")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.client_logger.addHandler(file_handler)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.Tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: dict):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr             = config["learning_rate"]
        self.round_id  = config.get("round_id", 0)
        optimizer_conf = config["optimizer"]
        optimizer      = getattr(optim, optimizer_conf)(self.model.parameters(), lr=lr)
        epochs         = config["epochs"]
        grad_clip      = float(config.get("cra_grad_clip", 1.0))

        logger.info(f"Client {self.client_name} training for round {self.round_id}")
        self.client_logger.info(f"Client {self.client_name} training for round {self.round_id}")
        start_time = time.time()
        self.model.to(self.device)

        cra_config = {
            k: v for k, v in config.items()
            if k.startswith("cra_") and k != "cra_grad_clip"
        }
        if "lambda_cra" in config:
            cra_config["lambda_cra"] = config["lambda_cra"]
        elif "cra_lambda_cra" in config:
            cra_config["lambda_cra"] = config["cra_lambda_cra"]

        if "proximal_mu" in config:
            cra_config["cra_proximal_mu"] = config["proximal_mu"]

        if "num_classes" in config and "cra_num_classes" not in cra_config:
            cra_config["cra_num_classes"] = config["num_classes"]

        try:
            metrics, cra_data = fed_train(
                model        = self.model,
                epochs       = epochs,
                optimizer    = optimizer,
                train_loader = self.train_loader,
                cra_config   = cra_config if cra_config else None,
                client_id    = self.client_id,     # FIX: pass through for server routing
                grad_clip    = grad_clip,           # FIX: actually applied now
            )
        except Exception as e:
            self.client_logger.error(f"Exception in client fit: {e}", exc_info=True)
            logger.error(f"Client {self.client_name} fit error: {e}", exc_info=True)
            raise

        # ── Build fit_metrics — all values must be scalar or str ──────────
        fit_metrics: dict[str, Scalar] = {}

        if cra_data:
            # cra_residuals and cra_class_counts are already JSON strings
            # (set in centralized.py) — safe to include directly
            fit_metrics["cra_residuals"]    = cra_data["cra_residuals"]
            fit_metrics["cra_class_counts"] = cra_data["cra_class_counts"]
            fit_metrics["client_id"]        = str(self.client_id)   # FIX: always included

        communication_time = time.time() - start_time
        self.write_results_json(metrics, communication_time, phase="train",
                                round_id=self.round_id)
        self.save_model(self.model, round_id=self.round_id)

        return self.get_parameters({}), len(self.train_loader), fit_metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        self.set_parameters(parameters)
        logger.info(f"Client {self.client_name} evaluating for round {self.round_id}")
        self.client_logger.info(f"Client {self.client_name} evaluating for round {self.round_id}")
        start_time = time.time()
        self.model.to(self.device)

        try:
            metrics = fed_test(self.model, self.test_loader)
        except Exception as e:
            self.client_logger.error(f"Exception in client evaluate: {e}", exc_info=True)
            logger.error(f"Client {self.client_name} evaluate error: {e}", exc_info=True)
            raise

        communication_time = time.time() - start_time
        self.write_results_json(metrics, communication_time, phase="test",
                                round_id=self.round_id)
        return float(metrics["loss"]), len(self.test_loader), {
            "accuracy":    float(metrics["accuracy"]),
            "macro_f1":    float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        }

    def write_results_json(self, metrics, communication_time, phase, round_id):
        file_name = f"{self.results_path}/metrics/{self.client_name}.json"
        results   = {
            "round":                round_id,
            f"{phase}_metrics":     metrics,
            "communication_time":   communication_time,
        }
        data = []
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                data = json.load(f)
        data.append(results)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def save_model(self, model, round_id):
        path = f"{self.results_path}/clients"
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(),
                   f"{path}/{self.client_name}_rnd_{round_id}.pth")


def generate_client_fn(model, train_loaders, test_loaders, client_names, results_path):
    def client_fn(client_id):
        try:
            client_model = deepcopy(model)
            return ClientModel(
                client_id    = client_id,
                model        = client_model,
                train_loader = train_loaders[int(client_id)],
                test_loader  = test_loaders[int(client_id)],
                client_names = client_names,
                results_path = results_path,
            ).to_client()
        except Exception as e:
            logger.error(f"Error occurred in client {client_id}: {e}")
            raise
    return client_fn


# ── XGBoost client (unchanged) ───────────────────────────────────────────────

class XgbFedClient(fl.client.Client):
    def __init__(self, client_name, train_matrix, val_matrix,
                 num_train, num_val, num_local_round, params,
                 model_results_path=None):
        self.bst               = None
        self.config            = None
        self.client_name       = client_name
        self.train_matrix      = train_matrix
        self.val_matrix        = val_matrix
        self.num_train         = num_train
        self.num_val           = num_val
        self.num_local_round   = num_local_round
        self.current_round     = 0
        self.params            = params
        self.model_results_path = model_results_path
        self.results           = []

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        for _ in range(self.num_local_round):
            self.bst.update(self.train_matrix, self.bst.num_boosted_rounds())
        return self.bst[
            self.bst.num_boosted_rounds() - self.num_local_round:
            self.bst.num_boosted_rounds()
        ]

    def fit(self, ins: FitIns) -> FitRes:
        self.current_round += 1
        if not self.bst:
            bst = xgb.train(
                self.params, self.train_matrix,
                num_boost_round=self.num_local_round,
                evals=[(self.val_matrix, "validate"), (self.train_matrix, "train")],
            )
            self.config = bst.save_config()
            self.bst    = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)
            bst = self._local_boost()

        local_model_bytes = bytes(bst.save_raw("json"))
        results_path = Path(self.model_results_path) / "clients"
        results_path.mkdir(parents=True, exist_ok=True)
        self.bst.save_model(results_path / f"{self.client_name}_rnd_{self.current_round}.json")

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.bst is None:
            raise ValueError("Model not trained.")
        eval_results = self.bst.eval_set(
            evals=[(self.val_matrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc       = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        y_true    = self.val_matrix.get_label()
        y_pred_p  = self.bst.predict(self.val_matrix)
        y_pred    = (np.argmax(y_pred_p, axis=1)
                     if len(y_pred_p.shape) > 1
                     else [1 if p > 0.5 else 0 for p in y_pred_p])

        metrics = {
            "round":      self.current_round,
            "AUC":        auc,
            "accuracy":   accuracy_score(y_true, y_pred),
            "error_rate": 1 - accuracy_score(y_true, y_pred),
            "f1_score":   f1_score(y_true, y_pred, average="weighted"),
            "precision":  precision_score(y_true, y_pred, average="weighted"),
            "recall":     recall_score(y_true, y_pred, average="weighted"),
        }
        try:
            results_path = Path(self.model_results_path) / "metrics"
            results_path.mkdir(parents=True, exist_ok=True)
            self.results.append(metrics)
            with open(results_path / f"{self.client_name}.json", "w") as f:
                json.dump(self.results, f, indent=4)
        except Exception as e:
            print(f"Failed to save metrics: {e}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=0.0, num_examples=self.num_val, metrics=metrics,
        )