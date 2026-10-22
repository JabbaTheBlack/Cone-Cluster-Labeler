#!/usr/bin/env python3
"""PointNet++ cone/non-cone detection model and training pipeline with early stopping."""

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


class PointNetSetAbstraction(nn.Module):
    """PointNet++-style local feature extraction using k-nearest neighbours."""

    def __init__(self, nsample, in_channel, mlp_channels):
        super().__init__()
        self.nsample = nsample

        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp_channels:
            layers.extend([
                nn.Conv2d(last_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ])
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, N, C)
        k = min(self.nsample, xyz.shape[1])
        distances = square_distance(xyz, xyz)
        indices = distances.topk(k, dim=-1, largest=False).indices

        batch_indices = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
        grouped_xyz = xyz[batch_indices, indices]
        grouped_xyz = grouped_xyz - xyz.unsqueeze(2)

        if points is None:
            grouped_features = grouped_xyz
        else:
            grouped_points = points[batch_indices, indices]
            grouped_features = torch.cat((grouped_xyz, grouped_points), dim=-1)

        grouped_features = grouped_features.permute(0, 3, 1, 2)
        features = self.mlp(grouped_features)
        features = features.max(dim=-1).values
        return features.permute(0, 2, 1)


def square_distance(src, dst):
    return (
        -2 * torch.matmul(src, dst.transpose(1, 2))
        + (src ** 2).sum(dim=-1).unsqueeze(2)
        + (dst ** 2).sum(dim=-1).unsqueeze(1)
    ).clamp_min(0.0)


class PointNet2(nn.Module):
    """PointNet++ classifier for input shaped (B, in_dim, N): x, y, z, (intensity)."""

    def __init__(self, num_classes=2, in_dim=4):
        super().__init__()
        self.in_dim = in_dim
        feature_dim = in_dim - 3

        self.sa1_small = PointNetSetAbstraction(6, feature_dim, [32, 64])
        self.sa1_large = PointNetSetAbstraction(16, feature_dim, [32, 64])
        self.sa2 = PointNetSetAbstraction(8, 128, [128, 256])
        self.fc1 = nn.Linear(256, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected input shape (B, {self.in_dim}, N), received {tuple(x.shape)}")

        x = x.transpose(1, 2)
        xyz = x[:, :, :3]
        features_in = x[:, :, 3:] if self.in_dim > 3 else None

        small = self.sa1_small(xyz, features_in)
        large = self.sa1_large(xyz, features_in)
        local_features = torch.cat((small, large), dim=-1)
        features = self.sa2(xyz, local_features)
        global_feature = features.max(dim=1).values
        output = F.relu(self.bn_fc1(self.fc1(global_feature)))
        return self.fc2(self.dropout(output))


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
    def __init__(self, raw_clouds, labels, num_points=32, augment=False, seed=42, use_intensity=True):
        self.num_points = num_points
        self.augment = augment
        self.use_intensity = use_intensity
        self.seed = seed
        self.labels = np.asarray(labels, dtype=np.int64)
        self.data = [self._preprocess(points, i, seed) for i, points in enumerate(raw_clouds)]

    def _preprocess(self, points, index, seed):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.zeros((4 if self.use_intensity else 3, self.num_points), dtype=np.float32)

        raw_xyz = points[:, :3]
        distances = np.linalg.norm(raw_xyz, axis=1, keepdims=True)
        xyz_centered = raw_xyz - raw_xyz.mean(axis=0, keepdims=True)

        if not self.use_intensity:
            normalized = xyz_centered
        else:
            raw_intensity = points[:, 3:4] if points.shape[1] >= 4 else np.zeros((len(points), 1), dtype=np.float32)
            i_normalized = np.clip(raw_intensity / 255.0, 0.0, 1.0)
            range_factor = (np.maximum(distances, 0.5) / 10.0) ** 2
            intensity = i_normalized * range_factor
            normalized = np.hstack((xyz_centered, intensity))

        # Seeded point sampler
        rng = np.random.default_rng(seed + index)
        sample_indices = rng.choice(len(normalized), self.num_points, replace=len(normalized) < self.num_points)
        sampled = normalized[sample_indices].T.astype(np.float32)

        if self.augment:
            sampled[:3] += rng.normal(0.0, 0.01, sampled[:3].shape).astype(np.float32)
            if self.use_intensity:
                sampled[3] *= rng.uniform(0.8, 1.2)

        return sampled

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.tensor(self.labels[index], dtype=torch.long)


class TrainingLogger:
    def __init__(self, repo_root, model_subfolder="pointnet2"):
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


def _save_confusion_matrix(y_true, y_pred, label_encoder, repo_root, model_name="pointnet2", output_name="pointnet2"):
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


def _balanced_group_split(y, groups, min_ratio=0.08, max_ratio=0.20):
    """Pick validation tracks whose size is within [min_ratio, max_ratio] of the samples and
    whose non-cone share best matches the remaining tracks (ties: more non-cones)."""
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


def run_pointnet2_pipeline(
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
    output_name="pointnet2",
    use_intensity=True,
):
    if len(raw_clouds) != len(y) or len(raw_clouds) == 0:
        raise ValueError("raw_clouds and y must be non-empty and have equal length")

    if epochs is None:
        epochs = 150
    epochs = int(epochs)

    # 1. Enforce global deterministic seeds
    seed_everything(random_state)

    if train_idx is None or test_idx is None:
        raise ValueError("train_idx and test_idx must be provided by the grouped dataset split.")

    raw_clouds = list(raw_clouds)
    y = np.asarray(y)
    groups = np.asarray(groups)

    fit_pos, val_pos = _balanced_group_split(y[train_idx], groups[train_idx])
    fit_idx, val_idx = np.asarray(train_idx)[fit_pos], np.asarray(train_idx)[val_pos]

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

    # Use worker seed generator for DataLoader shuffle reproducibility
    g = torch.Generator()
    g.manual_seed(random_state)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = 4 if use_intensity else 3
    model = PointNet2(num_classes=len(classes), in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(1.0 / np.maximum(counts, 1), device=device)
    weights /= weights.sum()
    criterion = FocalLoss(alpha=weights)

    print("\n[PointNet++ Detection Training]")
    print(f"  Device: {device}")
    print(f"  Train: {len(y_train)} ({train_counts})")
    print(f"  Val:   {len(y_val)} ({val_counts})  tracks: {sorted(set(groups[val_idx]))}")
    print(f"  Test:  {len(y_test)} ({test_counts})")
    print(f"  Early stopping: patience={patience}, min_delta={min_delta}")

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for points, target in train_loader:
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(points), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(target)
            
        avg_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, target in val_loader:
                points, target = points.to(device), target.to(device)
                loss = criterion(model(points), target)
                val_loss += loss.item() * len(target)
        val_loss /= len(val_dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:03d}/{epochs}: train_loss={avg_loss:.5f}, val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (best: {best_epoch + 1}, val_loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model from epoch {best_epoch + 1}")

    y_train_true, y_train_pred = _evaluate(model, train_eval_loader, device)
    y_test_true, y_test_pred = _evaluate(model, test_loader, device)
    metrics = {
        "model_type": "pointnet2",
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
        "best_val_loss": float(best_val_loss),
        "use_intensity": use_intensity,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch + 1,
        "num_points": num_points,
    }

    print("\n--- PointNet++ Detection Metrics ---")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Train F1 (Macro):{metrics['train_f1_macro']:.2%}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.2%}")
    print(f"  Precision:      {metrics['precision']:.2%}")
    print(f"  Recall:         {metrics['recall']:.2%}")
    print(f"  F1 Score (Cone):{metrics['f1_score']:.2%}")
    print(f"  F1 Score (Macro):{metrics['f1_macro']:.2%}")
    print(f"  Non-cone Recall:{metrics['noncone_recall']:.2%}")

    _save_confusion_matrix(y_test_true, y_test_pred, None, repo_root, model_name="pointnet2", output_name=output_name)
    save_dir = Path(repo_root) / "models" / "detection" / output_name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(classes), "num_points": num_points, "in_dim": in_dim, "use_intensity": use_intensity}, save_dir / "pointnet2_detection.pth")

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, in_dim, num_points)
    try:
        torch.onnx.export(model_cpu, dummy_input, save_dir / "pointnet2_detection.onnx", input_names=["input"], output_names=["output"], opset_version=17)
    except Exception as error:
        print(f"  ONNX export skipped: {error}")

    metrics["labels"] = [str(label) for label in classes]
    TrainingLogger(repo_root, model_subfolder=output_name).log_metrics(metrics)
    return model, metrics