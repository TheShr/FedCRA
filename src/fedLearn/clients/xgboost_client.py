import json
from pathlib import Path
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import flwr as fl
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