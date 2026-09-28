from pathlib import Path
from random import sample
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


class TrainingLogger:
    def __init__(self, repo_root: Path, model_subfolder: str):
        self.log_dir = Path(repo_root) / "logs" / "color" / model_subfolder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"


    def log_metrics(self, metrics: dict):
        with open(self.log_file, "a") as f:
            f.write("\n")
            f.write(json.dumps(metrics))


class MiniPointNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MiniPointNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)


        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)


        self.fc1 = nn.Linear(256, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))


        x = torch.max(x, 2)[0]


        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


class PointCloudDataset(Dataset):
    def __init__(self, raw_clouds, labels, num_points=32, augment=False):
        self.num_points = num_points
        self.augment = augment
        self.data = [self._preprocess(pts, num_points) for pts in raw_clouds]
        self.labels = labels


    def _preprocess(self, points, num_points):
        if len(points) == 0:
            return np.zeros((4, num_points), dtype=np.float32)


        xyz = points[:, :3] - points[:, :3].mean(axis=0)
        raw_intensity = points[:, 3:]


        i_min, i_max = raw_intensity.min(), raw_intensity.max()
        if i_max > i_min:
            intensity = (raw_intensity - i_min) / (i_max - i_min + 1e-8)
        else:
            intensity = np.zeros_like(raw_intensity)


        pts_norm = np.hstack([xyz, intensity])


        idx = np.random.choice(
            len(pts_norm),
            num_points,
            replace=(len(pts_norm) < num_points),
        )
        sampled_pts = pts_norm[idx].T.astype(np.float32)


        if self.augment:
            jitter = np.random.normal(
                0,
                0.01,
                size=sampled_pts[:3, :].shape,
            )
            sampled_pts[:3, :] += jitter
            sampled_pts[3, :] *= np.random.uniform(0.8, 1.2)


        return sampled_pts


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def visualize_confusion_matrix(y_true, y_pred, label_encoder, repo_root):
    cm = confusion_matrix(y_true, y_pred)
    labels = (
        list(label_encoder.classes_)
        if label_encoder
        else [str(i) for i in range(cm.shape[0])]
    )


    plt.figure(figsize=(7, 6))
    sns.set_context("notebook", font_scale=1.3)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14},
    )
    plt.title('Confusion Matrix - PointNet', fontsize=18, pad=12)
    plt.ylabel('True Color', fontsize=14)
    plt.xlabel('Predicted Color', fontsize=14)
    plt.tight_layout()


    out_dir = Path(repo_root) / 'figures' / 'color' / 'pointnet'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        out_dir / 'pointnet_confusion_matrix.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    print('✓ Saved: figures/color/pointnet/pointnet_confusion_matrix.png')


def _evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(device)
            outputs = model(pts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def run_pointnet_pipeline(
    raw_clouds,
    y,
    label_encoder,
    repo_root,
    epochs=150,
    train_idx=None,
    test_idx=None,
):
    raw_clouds = list(raw_clouds)
    y = np.asarray(y, dtype=np.int64)


    if train_idx is None or test_idx is None:
        raise ValueError(
            "PointNet requires precomputed leakage-safe "
            "group-disjoint train_idx and test_idx."
        )


    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)


    if np.intersect1d(train_idx, test_idx).size != 0:
        raise RuntimeError("Train/test sample-index overlap detected.")


    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/test indices must both be non-empty.")


    if train_idx.min() < 0 or test_idx.min() < 0:
        raise ValueError("Train/test indices must be non-negative.")


    if train_idx.max() >= len(raw_clouds) or test_idx.max() >= len(raw_clouds):
        raise ValueError("Train/test indices exceed dataset size.")


    X_train = [raw_clouds[index] for index in train_idx]
    X_test = [raw_clouds[index] for index in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]


    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "Grouped split must contain both classes in train and test."
        )


    classes = label_encoder.classes_
    train_counts = {
        str(cls): int((y_train == i).sum())
        for i, cls in enumerate(classes)
    }
    test_counts = {
        str(cls): int((y_test == i).sum())
        for i, cls in enumerate(classes)
    }


    train_str = " ".join(
        [f"{count} {cls}" for cls, count in train_counts.items()]
    )
    test_str = " ".join(
        [f"{count} {cls}" for cls, count in test_counts.items()]
    )


    print(f'\n[PointNet Training]')
    print(f'  Train: {len(X_train)} ({train_str})')
    print(f'  Test:  {len(X_test)} ({test_str})')


    train_ds = PointCloudDataset(X_train, y_train)
    test_ds = PointCloudDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniPointNet(num_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    class_counts = np.array(
        [train_counts[str(cls)] for cls in classes],
        dtype=np.float32,
    )
    class_weights = torch.tensor(
        1.0 / (class_counts / class_counts.sum()),
        device=device,
    )
    class_weights = class_weights / class_weights.sum()

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)


    print("\n--- Training Mini-PointNet ---")
    for epoch in range(epochs):
        model.train()

        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    y_train_true, y_train_pred = _evaluate_model(
        model,
        train_eval_loader,
        device,
    )
    y_test_true, y_test_pred = _evaluate_model(
        model,
        test_loader,
        device,
    )


    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(
        y_test_true,
        y_test_pred,
        average='macro',
        zero_division=0,
    )
    recall = recall_score(
        y_test_true,
        y_test_pred,
        average='macro',
        zero_division=0,
    )
    f1 = f1_score(
        y_test_true,
        y_test_pred,
        average='macro',
        zero_division=0,
    )


    print("\n--- PointNet Metrics ---")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"Precision:      {precision:.2%}")
    print(f"Recall:         {recall:.2%}")
    print(f"F1 Score:       {f1:.2%}")


    visualize_confusion_matrix(
        y_test_true,
        y_test_pred,
        label_encoder,
        repo_root,
    )


    save_dir = Path(repo_root) / 'models' / 'color' / 'pointnet'
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'pointnet_color.pth')


    dummy_input = torch.randn(1, 4, 32).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_dir / 'pointnet_color.onnx',
        input_names=['input'],
        output_names=['output'],
    )


    log_metrics = {
        "dataset_size": len(raw_clouds),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "epochs": epochs,
    }


    logger = TrainingLogger(repo_root, "pointnet")
    logger.log_metrics(log_metrics)