import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
from helpers.utils_ import to_tensor
import torch.optim as optim
from tqdm import tqdm

class CustomModule(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(CustomModule, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get the activation function from torch.nn based on the string name
        if isinstance(activation, str):
            try:
                self.activation = getattr(nn, activation)()
            except AttributeError:
                raise ValueError(f"Invalid activation function '{activation}'.")
        else:
            self.activation = activation

        self.fc_layers = nn.ModuleList()
        in_features = input_size
        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Linear(in_features, hidden_units))
            in_features = hidden_units
        self.fc_layers.append(nn.Linear(in_features, output_size))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            # Apply the activation function after each linear layer, except for the output layer
            if i < len(self.fc_layers) - 1:
                x = self.activation(x)
        return x

    def predict_proba(self, X_test):
        self.eval()
        X_test = to_tensor(X_test)
        dataset = TensorDataset(X_test)
        data_loader = DataLoader(dataset, batch_size=1)
        # Get the prediction probabilities
        prediction_probs = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self(inputs)
                probs = F.softmax(outputs, dim=1)
                prediction_probs.append(probs)
        return torch.cat(prediction_probs, dim=0).detach()

    def predict(self, X_test):
        self.eval()
        X_test = to_tensor(X_test)
        dataset = TensorDataset(X_test)
        data_loader = DataLoader(dataset, batch_size=1)
        predicted_labels = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self(inputs)
                predicted_label = torch.argmax(outputs, dim=1).item()
                predicted_labels.append(predicted_label)
        return predicted_labels

    def predict_shap(self, x):
        x = to_tensor(x)
        xx = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.exp(self.forward(xx))
        return probs.numpy()


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, hidden_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_neurons = hidden_units
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(input_size, hidden_units, hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_neurons).to(x.device)
        c0 = torch.zeros(self.hidden_layers, x.size(0), self.hidden_neurons).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class BaseModel:
    def __init__(self, model, epochs=10, batch_size = 32, learning_rate=0.001, verbose=True, criterion=None, optimizer=None):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
            raise TypeError("X_train and y_train must be torch.Tensor")

        if X_test is not None and y_test is not None:
            if not isinstance(X_test, torch.Tensor) or not isinstance(y_test, torch.Tensor):
                raise TypeError("X_test and y_test must be torch.Tensor")

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_test is not None and y_test is not None:
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

        # Wrap the epoch loop with tqdm
        epoch_progress = tqdm(range(self.epochs), desc="Training Progress", leave=True)
        for epoch in epoch_progress:
            # Training Phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_accuracy"].append(epoch_accuracy)

            # Validation Phase
            if X_test is not None and y_test is not None:
                self.model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)

                        # Compute test loss
                        loss = self.criterion(outputs, labels)
                        test_loss += loss.item()

                        # Compute test accuracy
                        _, predicted = torch.max(outputs, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()

                test_loss /= len(test_loader)
                test_accuracy = test_correct / test_total
                history["test_loss"].append(test_loss)
                history["test_accuracy"].append(test_accuracy)

                # Update tqdm with metrics
                epoch_progress.set_postfix({
                    "Train Loss": epoch_loss,
                    "Train Acc": epoch_accuracy,
                    "Test Loss": test_loss,
                    "Test Acc": test_accuracy,
                })

            else:
                # Update tqdm with train metrics only
                epoch_progress.set_postfix({
                    "Train Loss": epoch_loss,
                    "Train Acc": epoch_accuracy,
                })

        return history

    def predict(self, X_test, return_probabilities=False):
        if not isinstance(X_test, torch.Tensor):
            raise TypeError("X_test must be a torch.Tensor")
        self.model.eval()
        X_test = X_test.to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test)
            probabilities = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs, 1)
        if return_probabilities:
            return predicted, probabilities
        return predicted

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)







    

