#!/usr/bin/env python3
"""PointNet++ cone/non-cone detection model and training pipeline with early stopping."""



from pathlib import Path
import json
import random



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
    """PointNet++ classifier for input shaped (B, 4, N): x, y, z, intensity."""



    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1_small = PointNetSetAbstraction(6, 1, [32, 64])
        self.sa1_large = PointNetSetAbstraction(16, 1, [32, 64])
        self.sa2 = PointNetSetAbstraction(8, 128, [128, 256])
        self.fc1 = nn.Linear(256, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)



    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError(f"Expected input shape (B, 4, N), received {tuple(x.shape)}")



        x = x.transpose(1, 2)
        xyz, intensity = x[:, :, :3], x[:, :, 3:]
        small = self.sa1_small(xyz, intensity)
        large = self.sa1_large(xyz, intensity)
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
    def __init__(self, raw_clouds, labels, num_points=32, augment=False, seed=42):
        self.num_points = num_points
        self.augment = augment
        self.labels = np.asarray(labels, dtype=np.int64)
        self.data = [self._preprocess(points, i, seed) for i, points in enumerate(raw_clouds)]



    def _preprocess(self, points, index, seed):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.zeros((4, self.num_points), dtype=np.float32)



        if points.shape[1] >= 4:
            intensity = points[:, 3:4]
        else:
            intensity = np.zeros((len(points), 1), dtype=np.float32)



        xyz = points[:, :3] - points[:, :3].mean(axis=0, keepdims=True)
        i_min, i_max = intensity.min(), intensity.max()
        intensity = (intensity - i_min) / (i_max - i_min + 1e-8) if i_max > i_min else np.zeros_like(intensity)
        normalized = np.hstack((xyz, intensity))



        rng = np.random.default_rng(seed + index)
        sample_indices = rng.choice(len(normalized), self.num_points, replace=len(normalized) < self.num_points)
        sampled = normalized[sample_indices].T.astype(np.float32)



        if self.augment:
            sampled[:3] += rng.normal(0.0, 0.01, sampled[:3].shape).astype(np.float32)
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




def _save_confusion_matrix(y_true, y_pred, label_encoder, repo_root, model_name="pointnet2"):
    labels = list(label_encoder.classes_) if label_encoder is not None else ["0", "1"]
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    output_dir = Path(repo_root) / "figures" / "detection" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)



    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()} Detection")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()




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
):
    if len(raw_clouds) != len(y) or len(raw_clouds) == 0:
        raise ValueError("raw_clouds and y must be non-empty and have equal length")


    if epochs is None:
        epochs = 150
    epochs = int(epochs)



    if train_idx is None or test_idx is None:
        raise ValueError("train_idx and test_idx must be provided by the grouped dataset split.")


    raw_clouds = list(raw_clouds)
    y = np.asarray(y)
    groups = np.asarray(groups)


    clouds_train = [raw_clouds[index] for index in train_idx]
    clouds_test = [raw_clouds[index] for index in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]


    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError("Run leakage detected between train and test groups.")


    classes = np.unique(y)
    train_counts = {str(cls): int((y_train == cls).sum()) for cls in classes}
    test_counts = {str(cls): int((y_test == cls).sum()) for cls in classes}



    train_dataset = PointCloudDataset(clouds_train, y_train, num_points, augment=True, seed=random_state)
    test_dataset = PointCloudDataset(clouds_test, y_test, num_points, augment=False, seed=random_state)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(1.0 / np.maximum(counts, 1), device=device)
    weights /= weights.sum()
    criterion = FocalLoss(alpha=weights)



    print("\n[PointNet++ Detection Training]")
    print(f"  Device: {device}")
    print(f"  Train: {len(y_train)} ({train_counts})")
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
            for points, target in test_loader:
                points, target = points.to(device), target.to(device)
                loss = criterion(model(points), target)
                val_loss += loss.item() * len(target)
        val_loss /= len(test_dataset)


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
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_accuracy": float(accuracy_score(y_train_true, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test_true, y_test_pred)),
        "precision": float(precision_score(y_test_true, y_test_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)),
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch + 1,
        "num_points": num_points,
    }



    print("\n--- PointNet++ Detection Metrics ---")
    for key in ("train_accuracy", "test_accuracy", "precision", "recall", "f1_score"):
        print(f"  {key}: {metrics[key]:.2%}")



    _save_confusion_matrix(y_test_true, y_test_pred, None, repo_root, model_name="pointnet2")
    save_dir = Path(repo_root) / "models" / "detection" / "pointnet2"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(classes), "num_points": num_points}, save_dir / "pointnet2_detection.pth")



    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 4, num_points)
    torch.onnx.export(model_cpu, dummy_input, save_dir / "pointnet2_detection.onnx", input_names=["input"], output_names=["output"], opset_version=17)



    metrics["labels"] = [str(label) for label in classes]
    TrainingLogger(repo_root, model_subfolder="pointnet2").log_metrics(metrics)
    return model, metrics