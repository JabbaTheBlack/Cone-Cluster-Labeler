#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """PointNet classification network operating on raw point clusters (B, 3, N)."""
    def __init__(self, num_classes=2, in_dim=3):
        super(PointNetClassifier, self).__init__()
        self.tnet_input = TNet(k=in_dim)
        self.tnet_feature = TNet(k=64)

        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x shape: (B, in_dim, N)
        trans_input = self.tnet_input(x)
        x = torch.bmm(trans_input, x)

        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.tnet_feature(x)
        x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x


def run_pointnet_pipeline(*args, **kwargs):
    """Compatibility wrapper used by the unified detection trainer.

    The detection training script imports this symbol unconditionally and expects
    the same interface as the PointNet++ pipeline. Delegate to the implemented
    PointNet++ training pipeline so the older PointNet entry point remains usable.
    """
    from pointnet2_model import run_pointnet2_pipeline

    return run_pointnet2_pipeline(*args, **kwargs)


run_pointnet2_pipeline = run_pointnet_pipeline