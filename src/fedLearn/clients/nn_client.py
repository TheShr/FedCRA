"""
nn_client.py — Flower NumPyClient with FedCRA support.

CRA loss gating: the anchor loss for class k is only applied when the current
batch has at least MIN_CLASS_SAMPLES samples of class k. This prevents noisy
gradients from single-sample class estimates destabilising training.
"""

import torch.optim as optim
from collections import OrderedDict
from flwr.common import NDArrays
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
import os
import json
import time
from pathlib import Path
from log_config import base_logger
from src.fedLearn.centralized import fed_train, fed_test
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")

logger = base_logger(__name__)

# Minimum samples of a class in a mini-batch to compute a reliable centroid
MIN_CLASS_SAMPLES = 3


def _fed_train_cra(model, epochs, optimizer, train_loader,
                   anchors: np.ndarray, rho: np.ndarray, alpha_cra: float,
                   grad_clip: float = 1.0, class_weights: np.ndarray = None):
    """
    CE (class-balanced) + Class-Residual Anchoring loss.
    class_weights: per-class CE weights derived from minority severity.
    CRA term only pulls toward anchors with non-zero norm (initialised anchors).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    rho_t     = torch.tensor(rho,     dtype=torch.float32, device=device)
    K = anchors_t.shape[0]

    if class_weights is not None:
        w_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w_t)
    else:
        criterion = nn.CrossEntropyLoss()

    penultimate = []
    hook_handle = None
    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate.append(o)
        )
    else:
        logger.warning("FedCRA: penultimate layer not found — using CE only")
        return fed_train(model=model, epochs=epochs, optimizer=optimizer,
                         train_loader=train_loader)

    running_loss = 0.0
    steps = 0
    all_labels, all_preds = [], []

    try:
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                penultimate.clear()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_ce = criterion(outputs, labels)

                cra_terms = []
                if penultimate:
                    acts = penultimate[0]
                    for k in range(K):
                        mask = (labels == k)
                        # Gate: only use class k if enough samples for reliable centroid
                        if mask.sum() < MIN_CLASS_SAMPLES:
                            continue
                        mu_k = acts[mask].mean(dim=0)
                        # Skip uninitialised (all-zero) anchors
                        if anchors_t[k].norm().item() < 1e-6:
                            continue
                        diff = mu_k - anchors_t[k]
                        cra_terms.append(rho_t[k] * (diff * diff).sum())

                if cra_terms:
                    loss = loss_ce + alpha_cra * torch.stack(cra_terms).sum()
                else:
                    loss = loss_ce

                loss.backward()
                # Gradient clipping — prevents CRA loss spikes corrupting weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                running_loss += loss.item()
                steps += 1
                preds = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
    finally:
        if hook_handle:
            hook_handle.remove()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(K)))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    FPR = FP / (FP + TN + 1e-10)

    return {
        "loss":      running_loss / max(steps, 1),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_score":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_fpr": float(np.mean(FPR)),
    }


def _compute_class_centroids(model, loader, num_classes: int, device):
    """Compute per-class centroids from penultimate layer activations."""
    model.eval()
    sums, counts = {}, {}
    penultimate = []
    hook_handle = None

    if hasattr(model, "fc_layers") and len(model.fc_layers) >= 2:
        hook_handle = model.fc_layers[-2].register_forward_hook(
            lambda m, i, o: penultimate.append(o.detach())
        )
    else:
        return {}, {}

    try:
        with torch.no_grad():
            for xb, yb in loader:
                xb   = xb.to(device, non_blocking=True)
                yb_np = yb.numpy()
                penultimate.clear()
                _ = model(xb)
                if not penultimate:
                    continue
                acts = penultimate[0].cpu().numpy()
                for k in range(num_classes):
                    mask = (yb_np == k)
                    if mask.sum() == 0:
                        continue
                    k_acts = acts[mask]
                    if k not in sums:
                        sums[k]   = np.zeros(k_acts.shape[1], dtype=np.float32)
                        counts[k] = 0
                    sums[k]   += k_acts.sum(axis=0)
                    counts[k] += int(mask.sum())
    finally:
        if hook_handle:
            hook_handle.remove()

    return {k: (sums[k] / counts[k]).tolist() for k in sums}, counts


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
        self.client_name  = (client_names[int(client_id)]
                             if 0 <= int(client_id) < len(client_names)
                             else "client")
        self.model_name   = model.__class__.__name__

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.Tensor(v).to(self.device)
             for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: dict):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr            = config.get("learning_rate", 0.001)
        self.round_id = config.get("round_id", 0)
        optimizer     = getattr(optim, config.get("optimizer", "Adam"))(
                            self.model.parameters(), lr=lr)
        epochs        = config.get("epochs", 5)

        cra_anchors_json = config.get("cra_anchors")
        cra_rho_json     = config.get("cra_rho")
        alpha_cra        = float(config.get("cra_alpha", 0.0))
        grad_clip        = float(config.get("cra_grad_clip", 1.0))
        num_classes      = int(config.get("cra_num_classes",
                               getattr(self.model, "output_size", 9)))
        is_cra = bool(cra_anchors_json and cra_rho_json)

        print(f"Client {self.client_name} | round {self.round_id} | "
              f"lr={lr:.5f} | alpha={alpha_cra:.3f} | {'FedCRA' if is_cra else 'FedAvg'}")

        self.model.to(self.device)
        t0 = time.time()

        if is_cra:
            anchors = np.array(json.loads(cra_anchors_json), dtype=np.float32)
            rho     = np.array(json.loads(cra_rho_json),     dtype=np.float32)
            class_weights_json = config.get("cra_class_weights")
            class_weights = (np.array(json.loads(class_weights_json), dtype=np.float32)
                             if class_weights_json else None)
            metrics = _fed_train_cra(
                model=self.model, epochs=epochs, optimizer=optimizer,
                train_loader=self.train_loader,
                anchors=anchors, rho=rho, alpha_cra=alpha_cra,
                grad_clip=grad_clip, class_weights=class_weights)
        else:
            metrics = fed_train(
                model=self.model, epochs=epochs, optimizer=optimizer,
                train_loader=self.train_loader)

        self.write_results_json(metrics, time.time() - t0, "train", self.round_id)
        self.save_model(self.model, self.round_id)

        # Always send centroids when in CRA mode so anchors stay fresh
        fit_metrics = {}
        if is_cra:
            centroids, counts = _compute_class_centroids(
                self.model, self.train_loader, num_classes, self.device)
            fit_metrics["cra_centroids"] = json.dumps(
                {str(k): v for k, v in centroids.items()})
            fit_metrics["cra_counts"]    = json.dumps(
                {str(k): int(v) for k, v in counts.items()})

        return self.get_parameters({}), len(self.train_loader), fit_metrics

    def evaluate(self, parameters: NDArrays, config: dict):
        self.set_parameters(parameters)
        print(f"Client {self.client_name} evaluating round {self.round_id}")
        self.model.to(self.device)
        t0 = time.time()
        metrics = fed_test(self.model, self.test_loader)
        self.write_results_json(metrics, time.time() - t0, "test", self.round_id)
        return float(metrics["loss"]), len(self.test_loader), {"accuracy": metrics["accuracy"]}

    def write_results_json(self, metrics, dt, phase, round_id):
        d = Path(self.results_path) / "metrics"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{self.client_name}.json"
        data = json.loads(f.read_text()) if f.exists() else []
        data.append({"round": round_id, f"{phase}_metrics": metrics, "communication_time": dt})
        f.write_text(json.dumps(data, indent=4))

    def save_model(self, model, round_id):
        d = Path(self.results_path) / "clients"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / f"{self.client_name}_rnd_{round_id}.pth")


def generate_client_fn(model, train_loaders, test_loaders, client_names, results_path):
    def client_fn(client_id):
        try:
            return ClientModel(
                client_id=client_id, model=model,
                train_loader=train_loaders[int(client_id)],
                test_loader=test_loaders[int(client_id)],
                client_names=client_names,
                results_path=results_path,
            ).to_client()
        except Exception as e:
            logger.error(f"Error in client {client_id}: {e}")
            raise
    return client_fn
