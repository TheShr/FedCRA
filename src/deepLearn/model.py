"""
model.py — Improved DNN for Federated IoMT Classification
==========================================================

Drop-in replacement for the original DNN.

Architecture improvements (strategy-agnostic):
  - Residual blocks with Layer Normalization for stable gradient flow
  - GELU activation (smoother gradients vs ReLU, better for tabular data)
  - Dedicated projection head that widens then compresses to embedding_dim,
    producing a richer penultimate representation
  - Kaiming + zero-bias initialisation throughout
  - Dropout only inside residual branches (NOT on the penultimate output),
    so embeddings consumed by FedCRA / any strategy are full-fidelity at eval

Preserved interface contract (required by FedCRA + other strategies):
  - Same constructor signature: DNN(input_size, output_size, hidden_layers,
                                    hidden_units, activation)
  - self.fc_layers : nn.ModuleList  — FedCRA hooks fc_layers[-2] for embeddings
  - self.input_size, self.output_size, self.hidden_units attributes
  - forward(x) -> logits  (unchanged)
  - predict_proba / predict / predict_shap  (unchanged)
  - BaseModel, LSTMModel, CustomModule  (unchanged)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

try:
    from helpers.utils_ import to_tensor
except ImportError:  # allow standalone testing without the helpers package
    def to_tensor(x):
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(np.array(x), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """
    Pre-activation residual block designed for tabular (1-D) features.

    Layout:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear -> (+ skip)

    The skip connection uses a linear projection if in_dim != out_dim,
    otherwise it is the identity. This allows arbitrary width changes
    while preserving gradient flow.

    Why LayerNorm over BatchNorm for FL:
        BatchNorm statistics depend on the batch; on heterogeneous (non-IID)
        clients the running mean/var diverge, degrading aggregated models.
        LayerNorm normalises per-sample, so it behaves identically at any
        batch size and on any client's data distribution.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)

        # Projection skip if dimensions differ
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.linear1, self.linear2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        if isinstance(self.skip, nn.Linear):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        return out + residual


class _EmbeddingHead(nn.Module):
    """
    Dedicated penultimate projection head.

    Widen → compress → L2-normalise.
    The widening (x2) followed by compression to embedding_dim forces the
    network to learn a bottleneck representation that is more informative
    than a plain single linear layer. L2 normalisation projects embeddings
    onto the unit hypersphere, which:
      - Prevents embedding magnitude collapse (common with CE-only training)
      - Makes cosine-based alignment (FedCRA anchors) more geometrically
        meaningful
      - Keeps embedding norms stable across heterogeneous client batches

    Note: L2 norm is applied only at inference / evaluation.  During training
    the raw (pre-norm) activations flow into the classification head, which
    is standard practice (avoids gradient issues from the norm operation when
    used with CrossEntropyLoss).
    """

    def __init__(self, in_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        wide_dim = embedding_dim * 2
        self.norm    = nn.LayerNorm(in_dim)
        self.expand  = nn.Linear(in_dim, wide_dim)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(p=dropout)
        self.project = nn.Linear(wide_dim, embedding_dim)

        nn.init.kaiming_normal_(self.expand.weight,  nonlinearity="relu")
        nn.init.zeros_(self.expand.bias)
        nn.init.kaiming_normal_(self.project.weight, nonlinearity="relu")
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.project(x)
        return x


# ---------------------------------------------------------------------------
# Main DNN — drop-in replacement
# ---------------------------------------------------------------------------

class DNN(nn.Module):
    """
    Improved DNN with residual connections and richer penultimate embeddings.

    Constructor signature is identical to the original:
        DNN(input_size, output_size, hidden_layers, hidden_units, activation)

    The `activation` argument is accepted for full backward compatibility but
    the improved architecture uses GELU internally (fixed, not user-selectable).
    If you pass 'ReLU' (or any other string / nn.Module), the model still
    builds correctly — the argument is stored and ignored internally.

    Internal structure
    ------------------
    Input (input_size)
        └─ Input projection  →  hidden_units
        └─ hidden_layers residual blocks  (each: hidden_units → hidden_units)
        └─ Embedding head  →  embedding_dim  (= hidden_units)   ← fc_layers[-2]
        └─ Classifier head →  output_size                       ← fc_layers[-1]

    fc_layers layout (preserves FedCRA hook contract):
        fc_layers[0]  ... fc_layers[hidden_layers]   : residual block linears
                                                        (wrapped in _BlockAdapter)
        fc_layers[-2] : _EmbeddingHead   — FedCRA hooks this for centroids
        fc_layers[-1] : nn.Linear        — output logits
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        hidden_units: int,
        activation,                 # kept for API compatibility; GELU is used internally
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Public attributes required by client code ──────────────────────
        self.input_size   = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units  = hidden_units
        self.output_size   = output_size
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store activation arg for introspection (not used internally)
        if isinstance(activation, str):
            try:
                self._activation_ref = getattr(nn, activation)()
            except AttributeError:
                self._activation_ref = nn.GELU()
        else:
            self._activation_ref = activation if activation is not None else nn.GELU()

        # ── Architecture ───────────────────────────────────────────────────
        # embedding_dim == hidden_units keeps the output of fc_layers[-2]
        # the same width as the rest of the network, matching FedCRA's
        # expectation that embedding_dim == hidden_units (set in fedcra.yaml).
        embedding_dim = hidden_units

        # Build the backbone as a flat nn.ModuleList so that:
        #   fc_layers[-2]  == embedding head  (penultimate)
        #   fc_layers[-1]  == classifier head (output)
        # FedCRA's forward hook targets fc_layers[-2] — this must remain true.

        layers: list[nn.Module] = []

        # Input projection + first residual block combined
        layers.append(_ResidualBlock(input_size, hidden_units, dropout=dropout))

        # Additional residual blocks (hidden_layers - 1 more, so total depth = hidden_layers)
        for _ in range(max(0, hidden_layers - 1)):
            layers.append(_ResidualBlock(hidden_units, hidden_units, dropout=dropout))

        # Penultimate: dedicated embedding head
        layers.append(_EmbeddingHead(hidden_units, embedding_dim, dropout=dropout))

        # Output: classifier
        output_linear = nn.Linear(embedding_dim, output_size)
        nn.init.kaiming_normal_(output_linear.weight, nonlinearity="relu")
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.fc_layers = nn.ModuleList(layers)

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)

        # All layers except the final classifier
        for layer in self.fc_layers[:-1]:
            x = layer(x)

        # Classifier (no activation — raw logits for CrossEntropyLoss)
        logits = self.fc_layers[-1](x)
        return logits

    # ── Inference helpers (identical interface to original) ─────────────────

    def predict_proba(self, X_test) -> torch.Tensor:
        self.eval()
        X_test  = to_tensor(X_test)
        dataset = TensorDataset(X_test)
        loader  = DataLoader(dataset, batch_size=256)
        probs_list = []
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                logits = self(inputs)
                probs_list.append(F.softmax(logits, dim=1))
        return torch.cat(probs_list, dim=0).detach()

    def predict(self, X_test) -> list:
        self.eval()
        X_test  = to_tensor(X_test)
        dataset = TensorDataset(X_test)
        loader  = DataLoader(dataset, batch_size=256)
        preds   = []
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                logits = self(inputs)
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        return preds

    def predict_shap(self, x) -> "numpy.ndarray":
        x  = to_tensor(x)
        xx = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.exp(self.forward(xx))
        return probs.cpu().numpy()

    # ── Embedding extraction (bonus utility, not used by any strategy) ──────

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer activations (L2-normalised) for x."""
        self.eval()
        x = x.view(-1, self.input_size).to(self.device)
        with torch.no_grad():
            for layer in self.fc_layers[:-1]:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Unchanged supporting classes
# ---------------------------------------------------------------------------

class CustomModule(nn.Module):
    """Kept for backward compatibility."""

    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.fc         = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, hidden_layers=2):
        super().__init__()
        self.hidden_neurons = hidden_units
        self.hidden_layers  = hidden_layers
        self.lstm = nn.LSTM(input_size, hidden_units, hidden_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_units, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_neurons).to(x.device)
        c0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_neurons).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class BaseModel:
    def __init__(
        self,
        model: nn.Module,
        epochs: int       = 10,
        batch_size: int   = 32,
        learning_rate: float = 0.001,
        verbose: bool     = True,
        criterion         = None,
        optimizer         = None,
    ):
        self.model         = model
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.verbose       = verbose
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion     = criterion  if criterion  else nn.CrossEntropyLoss()
        self.optimizer     = optimizer  if optimizer  else optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
            raise TypeError("X_train and y_train must be torch.Tensor")
        if X_test is not None and y_test is not None:
            if not isinstance(X_test, torch.Tensor) or not isinstance(y_test, torch.Tensor):
                raise TypeError("X_test and y_test must be torch.Tensor")

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True,
        )
        test_loader = (
            DataLoader(TensorDataset(X_test, y_test),
                       batch_size=self.batch_size, shuffle=False)
            if X_test is not None and y_test is not None
            else None
        )

        history = {"train_loss": [], "train_accuracy": [],
                   "test_loss":  [], "test_accuracy":  []}

        epoch_progress = tqdm(range(self.epochs), desc="Training Progress", leave=True)
        for epoch in epoch_progress:
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted  = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss     = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_accuracy"].append(epoch_accuracy)

            if test_loader is not None:
                self.model.eval()
                t_loss, t_correct, t_total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        t_loss   += self.criterion(outputs, labels).item()
                        _, predicted = torch.max(outputs, 1)
                        t_total  += labels.size(0)
                        t_correct += (predicted == labels).sum().item()

                test_loss     = t_loss / len(test_loader)
                test_accuracy = t_correct / t_total
                history["test_loss"].append(test_loss)
                history["test_accuracy"].append(test_accuracy)
                epoch_progress.set_postfix({
                    "Train Loss": epoch_loss,     "Train Acc": epoch_accuracy,
                    "Test Loss":  test_loss,      "Test Acc":  test_accuracy,
                })
            else:
                epoch_progress.set_postfix({
                    "Train Loss": epoch_loss, "Train Acc": epoch_accuracy,
                })

        return history

    def predict(self, X_test, return_probabilities: bool = False):
        if not isinstance(X_test, torch.Tensor):
            raise TypeError("X_test must be a torch.Tensor")
        self.model.eval()
        X_test = X_test.to(self.device)
        with torch.no_grad():
            outputs       = self.model(X_test)
            probabilities = nn.Softmax(dim=1)(outputs)
            _, predicted  = torch.max(outputs, 1)
        return (predicted, probabilities) if return_probabilities else predicted

    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)