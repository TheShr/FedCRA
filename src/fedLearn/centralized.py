import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


def get_gpu_memory_info(device=None):
    """Get detailed GPU memory information."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "available_gb": max(0, total - reserved)
        }
    return None


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(stage: str = "", device=None):
    """Log GPU memory usage at different stages."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        mem_info = get_gpu_memory_info(device)
        if mem_info:
            msg = f"{stage} - GPU Memory: {mem_info['allocated_gb']:.2f}/{mem_info['total_gb']:.2f} GB"
            print(msg)


def compute_macro_fpr(labels, predictions, num_classes=None):
    labels_list = list(labels)
    preds_list  = list(predictions)
    if num_classes:
        lab_set = list(range(num_classes))
    else:
        lab_set = sorted(set(labels_list) | set(preds_list))
    cm = confusion_matrix(labels_list, preds_list, labels=lab_set)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    FPR = FP / (FP + TN + 1e-10)
    return float(np.mean(FPR))


def compute_per_class_f1(labels, predictions, num_classes=None):
    """Return dict {class_id: f1} for every class."""
    if num_classes:
        lab_set = list(range(num_classes))
    else:
        lab_set = sorted(set(list(labels)) | set(list(predictions)))
    f1_per = f1_score(labels, predictions, labels=lab_set,
                      average=None, zero_division=0)
    return {int(k): float(v) for k, v in zip(lab_set, f1_per)}


def fed_train(model, epochs, optimizer, train_loader, class_weights=None,
              grad_clip: float = 1.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log_gpu_memory("Train start", device)

    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=w)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.train()
    running_loss = 0.0
    epoch_steps  = 0
    total = 0
    correct = 0
    all_labels, all_predictions = [], []

    _clip = float(grad_clip) if grad_clip and grad_clip > 0 else float("inf")

    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # ✅ better performance

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            if _clip < float("inf"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_clip)

            optimizer.step()

            running_loss += loss.item()
            epoch_steps  += 1

            # ✅ removed .data
            _, predicted = torch.max(outputs, 1)

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            # (kept same behavior, minimal change)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    log_gpu_memory("Train end", device)
    if device.type == "cuda":
        clear_gpu_cache()

    return {
        "loss":        running_loss / max(epoch_steps, 1),
        "accuracy":    correct / max(total, 1),
        "precision":   precision_score(all_labels, all_predictions,
                                       average="macro", zero_division=0),
        "recall":      recall_score(all_labels, all_predictions,
                                    average="macro", zero_division=0),
        "f1_score":    f1_score(all_labels, all_predictions,
                                 average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_predictions,
                                 average="weighted", zero_division=0),
        "macro_fpr":   compute_macro_fpr(all_labels, all_predictions),
    }


def fed_test(model, test_loader, num_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log_gpu_memory("Test start", device)
    
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    total_samples = 0
    total_loss = 0.0
    all_labels, all_predictions = [], []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # ✅ FIX: correct loss scaling
            total_loss += loss.item() * labels.size(0)

            # ✅ removed .data
            _, predicted = torch.max(outputs, 1)

            correct       += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    log_gpu_memory("Test end", device)
    if device.type == "cuda":
        clear_gpu_cache()

    metrics = {
        "loss":         total_loss / max(total_samples, 1),  # ✅ now correct
        "accuracy":     accuracy_score(all_labels, all_predictions),
        "precision":    precision_score(all_labels, all_predictions,
                                        average="macro", zero_division=0),
        "recall":       recall_score(all_labels, all_predictions,
                                     average="macro", zero_division=0),
        "f1_score":     f1_score(all_labels, all_predictions,
                                  average="macro", zero_division=0),
        "f1_weighted":  f1_score(all_labels, all_predictions,
                                  average="weighted", zero_division=0),
        "macro_fpr":    compute_macro_fpr(all_labels, all_predictions,
                                           num_classes=num_classes),
        "per_class_f1": compute_per_class_f1(all_labels, all_predictions,
                                              num_classes=num_classes),
    }

    return metrics