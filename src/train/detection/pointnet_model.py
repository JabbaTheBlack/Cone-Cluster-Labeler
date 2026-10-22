#!/usr/bin/env python3
"""PointNet cone/non-cone detection model with early stopping."""
from pathlib import Path
import json
import random
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def seed_everything(seed=42):
    """Ensure complete reproducibility across Python, NumPy, and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithm choices in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TNet(nn.Module):
    """Spatial Transformer Network (T-Net) for point alignment."""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        init = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        matrix = self.fc3(x) + init
        return matrix.view(-1, self.k, self.k)


class PointNetClassifier(nn.Module):
    """PointNet classification network operating on raw point clusters (B, in_dim, N).

    arch="modified": 4x4 input T-Net over x, y, z and intensity.
    arch="paper": 3x3 input T-Net on x, y, z only, two 64-wide layers before feature T-Net."""
    def __init__(self, num_classes=2, in_dim=4, arch="modified", dropout=0.3):
        super(PointNetClassifier, self).__init__()
        if arch not in ("modified", "paper"):
            raise ValueError(f"Unknown PointNet arch: {arch}")
        self.arch = arch
        self.tnet_input = TNet(k=3 if arch == "paper" else in_dim)
        self.tnet_feature = TNet(k=64)

        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        if arch == "paper":
            self.conv1b = nn.Conv1d(64, 64, 1)
            self.bn1b = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.trans_feat = None

    def forward(self, x):
        if self.arch == "paper":
            xyz = torch.bmm(self.tnet_input(x[:, :3]), x[:, :3])
            x = torch.cat([xyz, x[:, 3:]], dim=1)
        else:
            trans_input = self.tnet_input(x)
            x = torch.bmm(trans_input, x)

        x = F.relu(self.bn1(self.conv1(x)))
        if self.arch == "paper":
            x = F.relu(self.bn1b(self.conv1b(x)))

        trans_feat = self.tnet_feature(x)
        self.trans_feat = trans_feat
        x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x


def feature_transform_regularizer(trans):
    """Qi et al. 2017: ||I - A A^T||_F^2 keeps the 64x64 feature transform close to orthogonal."""
    identity = torch.eye(trans.size(1), device=trans.device).unsqueeze(0)
    return torch.mean(torch.norm(identity - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2)) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        focal = (1.0 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


class PointCloudDataset(Dataset):
    """Centred xyz + range-power normalized intensity, resampled to num_points on every access."""

    def __init__(self, raw_clouds, labels, num_points=32, augment=False, seed=42, use_intensity=True):
        self.num_points = num_points
        self.augment = augment
        self.use_intensity = use_intensity
        self.seed = seed
        self.labels = np.asarray(labels, dtype=np.int64)
        self.data = [self._normalize(points) for points in raw_clouds]
        self.rng = np.random.default_rng(seed)

    def _normalize(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.zeros((1, 4 if self.use_intensity else 3), dtype=np.float32)

        raw_xyz = points[:, :3]
        distances = np.linalg.norm(raw_xyz, axis=1, keepdims=True)
        xyz_centered = raw_xyz - raw_xyz.mean(axis=0, keepdims=True)

        if not self.use_intensity:
            return xyz_centered.astype(np.float32)

        if points.shape[1] >= 4:
            raw_intensity = points[:, 3:4]
        else:
            raw_intensity = np.zeros((len(points), 1), dtype=np.float32)

        i_normalized = np.clip(raw_intensity / 255.0, 0.0, 1.0)
        range_factor = (np.maximum(distances, 0.5) / 10.0) ** 2
        intensity = i_normalized * range_factor

        return np.hstack((xyz_centered, intensity)).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.data[index]
        rng = self.rng if self.augment else np.random.default_rng(self.seed + index)
        if len(points) >= self.num_points:
            sample_indices = rng.choice(len(points), self.num_points, replace=False)
        else:
            sample_indices = np.concatenate([
                np.arange(len(points)),
                rng.choice(len(points), self.num_points - len(points))
            ])
        sampled = points[sample_indices].T.copy()

        if self.augment:
            sampled[:3] += rng.normal(0.0, 0.01, sampled[:3].shape).astype(np.float32)
            if self.use_intensity:
                sampled[3] *= rng.uniform(0.8, 1.2)

        return torch.from_numpy(sampled), torch.tensor(self.labels[index], dtype=torch.long)


class TrainingLogger:
    def __init__(self, repo_root, model_subfolder="pointnet"):
        self.log_dir = Path(repo_root) / "logs" / "detection" / model_subfolder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"

    def log_metrics(self, metrics):
        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")


def _evaluate(model, loader, device):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for points, target in loader:
            output = model(points.to(device))
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            labels.extend(target.numpy())
    return np.asarray(labels), np.asarray(predictions)


def _save_confusion_matrix(y_true, y_pred, label_encoder, repo_root, model_name="pointnet", output_name="pointnet"):
    labels = list(label_encoder.classes_) if label_encoder is not None else ["0", "1"]
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    output_dir = Path(repo_root) / "figures" / "detection" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()} Detection")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def train_pointnet(train_dataset, val_dataset, num_classes, in_dim, device, arch="modified", dropout=0.3,
                   epochs=150, batch_size=32, learning_rate=1e-3, lr_step=0, weight_decay=0.0,
                   loss="focal", reg_weight=0.0, patience=15, min_delta=1e-4, verbose=True, seed=42):
    """Train one PointNet model with early stopping."""
    model = PointNetClassifier(num_classes=num_classes, in_dim=in_dim, arch=arch, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5) if lr_step else None
    counts = np.bincount(train_dataset.labels, minlength=num_classes).astype(np.float32)
    weights = torch.tensor(1.0 / np.maximum(counts, 1), device=device)
    weights /= weights.sum()
    criterion = FocalLoss(alpha=weights) if loss == "focal" else nn.CrossEntropyLoss(weight=weights * num_classes)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None
    best_val_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for points, target in train_loader:
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = criterion(model(points), target)
            if reg_weight:
                batch_loss = batch_loss + reg_weight * feature_transform_regularizer(model.trans_feat)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item() * len(target)
        if scheduler is not None:
            scheduler.step()
        avg_loss = running_loss / len(train_dataset)

        if val_loader is None:
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs):
                print(f"  Epoch {epoch + 1:03d}/{epochs}: train_loss={avg_loss:.5f}")
            best_epoch = epoch
            continue

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device), target.to(device)
                val_loss += criterion(model(points), target).item() * len(target)
        val_loss /= len(val_dataset)
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch + 1:03d}/{epochs}: train_loss={avg_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} (best: {best_epoch + 1}, val_loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Restored best model from epoch {best_epoch + 1}")
    return model, best_epoch, best_val_loss, epoch + 1


def _balanced_group_split(y, groups, min_ratio=0.08, max_ratio=0.20):
    """Pick validation tracks whose size is within [min_ratio, max_ratio] of samples."""
    unique_groups = np.unique(groups)
    best = None
    for k in range(1, len(unique_groups)):
        for combo in combinations(unique_groups, k):
            is_val = np.isin(groups, combo)
            ratio = is_val.mean()
            if not (min_ratio <= ratio <= max_ratio) or len(np.unique(y[is_val])) < 2 or len(np.unique(y[~is_val])) < 2:
                continue
            score = (abs(y[is_val].mean() - y[~is_val].mean()), -int((y[is_val] == 0).sum()))
            if best is None or score < best[0]:
                best = (score, is_val)
    if best is None:
        raise RuntimeError("No validation track combination satisfies the size and class constraints.")
    is_val = best[1]
    return np.where(~is_val)[0], np.where(is_val)[0]


def run_pointnet_pipeline(
    raw_clouds,
    y,
    groups,
    repo_root,
    epochs=150,
    num_points=32,
    batch_size=32,
    learning_rate=1e-3,
    random_state=42,
    patience=15,
    min_delta=1e-4,
    train_idx=None,
    test_idx=None,
    output_name="pointnet",
    use_validation=True,
    use_intensity=True,
    arch="modified",
    dropout=0.3,
    lr_step=0,
    weight_decay=0.0,
    loss="focal",
    reg_weight=0.0,
):
    """Train, evaluate, log and export PointNet detection model deterministically."""
    if len(raw_clouds) != len(y) or len(raw_clouds) == 0:
        raise ValueError("raw_clouds and y must be non-empty and have equal length")

    if epochs is None:
        epochs = 150
    epochs = int(epochs)

    # Enforce global deterministic seeds
    seed_everything(random_state)

    if train_idx is None or test_idx is None:
        raise ValueError("train_idx and test_idx must be provided by the grouped dataset split.")

    raw_clouds = list(raw_clouds)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if use_validation:
        fit_pos, val_pos = _balanced_group_split(y[train_idx], groups[train_idx])
        fit_idx, val_idx = np.asarray(train_idx)[fit_pos], np.asarray(train_idx)[val_pos]
    else:
        fit_idx, val_idx = np.asarray(train_idx), np.asarray([], dtype=int)

    clouds_train = [raw_clouds[index] for index in fit_idx]
    clouds_val = [raw_clouds[index] for index in val_idx]
    clouds_test = [raw_clouds[index] for index in test_idx]
    y_train = y[fit_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    for a, b in [(fit_idx, val_idx), (fit_idx, test_idx), (val_idx, test_idx)]:
        if not set(groups[a]).isdisjoint(set(groups[b])):
            raise RuntimeError("Run leakage detected between train/validation/test groups.")

    classes = np.unique(y)
    train_counts = {str(cls): int((y_train == cls).sum()) for cls in classes}
    val_counts = {str(cls): int((y_val == cls).sum()) for cls in classes}
    test_counts = {str(cls): int((y_test == cls).sum()) for cls in classes}

    train_dataset = PointCloudDataset(clouds_train, y_train, num_points, augment=True, seed=random_state, use_intensity=use_intensity)
    train_eval_dataset = PointCloudDataset(clouds_train, y_train, num_points, augment=False, seed=random_state, use_intensity=use_intensity)
    val_dataset = PointCloudDataset(clouds_val, y_val, num_points, augment=False, seed=random_state, use_intensity=use_intensity)
    test_dataset = PointCloudDataset(clouds_test, y_test, num_points, augment=False, seed=random_state, use_intensity=use_intensity)

    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = 4 if use_intensity else 3

    print("\n[PointNet Detection Training]")
    print(f"  Device: {device} | arch: {arch} | input channels: {'x, y, z, intensity' if use_intensity else 'x, y, z'} | seed: {random_state}")
    print(f"  Train: {len(y_train)} ({train_counts})")
    print(f"  Val:   {len(y_val)} ({val_counts})  tracks: {sorted(set(groups[val_idx]))}")
    print(f"  Test:  {len(y_test)} ({test_counts})")
    if use_validation:
        print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
    else:
        print(f"  No validation: training on all training tracks for {epochs} epochs")

    model, best_epoch, best_val_loss, epochs_trained = train_pointnet(
        train_dataset, val_dataset if use_validation else None, num_classes=len(classes), in_dim=in_dim,
        device=device, arch=arch, dropout=dropout, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, lr_step=lr_step, weight_decay=weight_decay, loss=loss,
        reg_weight=reg_weight, patience=patience, min_delta=min_delta, seed=random_state
    )
    epoch = epochs_trained - 1

    y_train_true, y_train_pred = _evaluate(model, train_eval_loader, device)
    y_test_true, y_test_pred = _evaluate(model, test_loader, device)
    metrics = {
        "model_type": "pointnet",
        "task": "detection",
        "dataset_size": len(raw_clouds),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "val_size": len(y_val),
        "val_tracks": sorted(set(groups[val_idx])),
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "test_class_counts": test_counts,
        "train_accuracy": float(accuracy_score(y_train_true, y_train_pred)),
        "train_f1_macro": float(f1_score(y_train_true, y_train_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test_true, y_test_pred)),
        "precision": float(precision_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)),
        "noncone_precision": float(precision_score(y_test_true, y_test_pred, pos_label=0, zero_division=0)),
        "noncone_recall": float(recall_score(y_test_true, y_test_pred, pos_label=0, zero_division=0)),
        "noncone_f1_score": float(f1_score(y_test_true, y_test_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test_true, y_test_pred, labels=[0, 1]).tolist(),
        "best_val_loss": float(best_val_loss) if use_validation else None,
        "use_validation": use_validation,
        "use_intensity": use_intensity,
        "seed": random_state,
        "arch": arch,
        "hyperparameters": {"learning_rate": learning_rate, "lr_step": lr_step, "dropout": dropout,
                            "weight_decay": weight_decay, "loss": loss, "reg_weight": reg_weight,
                            "patience": patience, "batch_size": batch_size},
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch + 1,
        "num_points": num_points,
    }

    print("\n--- PointNet Detection Metrics ---")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Train F1 (Macro):{metrics['train_f1_macro']:.2%}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.2%}")
    print(f"  Precision:      {metrics['precision']:.2%}")
    print(f"  Recall:         {metrics['recall']:.2%}")
    print(f"  F1 Score (Cone):{metrics['f1_score']:.2%}")
    print(f"  F1 Score (Macro):{metrics['f1_macro']:.2%}")
    print(f"  Non-cone Recall:{metrics['noncone_recall']:.2%}")

    _save_confusion_matrix(y_test_true, y_test_pred, None, repo_root, model_name="pointnet", output_name=output_name)
    save_dir = Path(repo_root) / "models" / "detection" / output_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(classes), "num_points": num_points,
                "in_dim": in_dim, "use_intensity": use_intensity, "arch": arch}, save_dir / "pointnet_detection.pth")

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, in_dim, num_points)
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            save_dir / "pointnet_detection.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=False,
        )
    except Exception as error:
        print(f"  ONNX export skipped: {error}")

    metrics["labels"] = [str(label) for label in classes]
    TrainingLogger(repo_root, model_subfolder=output_name).log_metrics(metrics)
    return model, metrics