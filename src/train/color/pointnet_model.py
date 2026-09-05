from pathlib import Path
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

        x = torch.max(x, 2)[0]  # Global max pooling

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


class PointCloudDataset(Dataset):
    def __init__(self, raw_clouds, labels, num_points=32):
        self.data = [self._preprocess(pts, num_points) for pts in raw_clouds]
        self.labels = labels

    def _preprocess(self, points, num_points):
        if len(points) == 0:
            return np.zeros((4, num_points), dtype=np.float32)
        xyz = points[:, :3] - points[:, :3].mean(axis=0)
        intensity = points[:, 3:]
        
        # Normalize intensity to range [0, 1] if non-empty
        if intensity.max() > intensity.min():
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            
        pts_norm = np.hstack([xyz, intensity])
        idx = np.random.choice(len(pts_norm), num_points, replace=(len(pts_norm) < num_points))
        return pts_norm[idx].T.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def visualize_confusion_matrix(y_true, y_pred, label_encoder, repo_root):
    cm = confusion_matrix(y_true, y_pred)
    labels = list(label_encoder.classes_) if label_encoder else [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(7, 6))
    sns.set_context("notebook", font_scale=1.3)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 14}
    )
    plt.title('Confusion Matrix - PointNet', fontsize=18, pad=12)
    plt.ylabel('True Color', fontsize=14)
    plt.xlabel('Predicted Color', fontsize=14)
    plt.tight_layout()

    out_dir = Path(repo_root) / 'figures' / 'color' / 'pointnet'
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'pointnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
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


def run_pointnet_pipeline(raw_clouds, y, label_encoder, repo_root, epochs=40):

    X_train, X_test, y_train, y_test = train_test_split(
        raw_clouds, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = label_encoder.classes_
    train_counts = {str(cls): int((y_train == i).sum()) for i, cls in enumerate(classes)}
    test_counts = {str(cls): int((y_test == i).sum()) for i, cls in enumerate(classes)}

    train_str = " ".join([f"{count} {cls}" for cls, count in train_counts.items()])
    test_str = " ".join([f"{count} {cls}" for cls, count in test_counts.items()])

    print(f'\n[PointNet Training]')
    print(f'  Train: {len(X_train)} ({train_str})')
    print(f'  Test:  {len(X_test)} ({test_str})')

    train_ds = PointCloudDataset(X_train, y_train)

    train_ds = PointCloudDataset(X_train, y_train)
    test_ds = PointCloudDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniPointNet(num_classes=len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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

    # Model Evaluation
    y_train_true, y_train_pred = _evaluate_model(model, train_eval_loader, device)
    y_test_true, y_test_pred = _evaluate_model(model, test_loader, device)

    train_acc = accuracy_score(y_train_true, y_train_pred)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)

    print("\n--- PointNet Metrics ---")
    print(f"Train Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")
    print(f"Precision:      {precision:.2%}")
    print(f"Recall:         {recall:.2%}")
    print(f"F1 Score:       {f1:.2%}")

    visualize_confusion_matrix(y_test_true, y_test_pred, label_encoder, repo_root)

    save_dir = Path(repo_root) / 'models' / 'color' / 'pointnet'
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'pointnet_color.pth')

    dummy_input = torch.randn(1, 4, 32).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        save_dir / 'pointnet_color.onnx',
        input_names=['input'], 
        output_names=['output']
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
        "epochs": epochs
    }

    logger = TrainingLogger(repo_root, "pointnet")
    logger.log_metrics(log_metrics)