import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.teacher import TeacherSegmentationModel
from training.dataset import LidarScanDataset

def train_teacher(data_dir: str, save_path: str = "models/teacher.pt", epochs: int = 2, lr: float = 0.001):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = LidarScanDataset(data_dir)

    if len(dataset) == 0:
        print(f"[Train Teacher] Warning: No dataset scans found in {data_dir}. Creating initialized model weights at {save_path}.")
        model = TeacherSegmentationModel()
        torch.save(model.state_dict(), save_path)
        return model

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = TeacherSegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[Train Teacher] Training Teacher Model over {len(dataset)} scans for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for sample in dataloader:
            pts = sample["points"][0]
            feats = sample["features"][0]
            labels = sample["labels"][0]

            optimizer.zero_grad()
            logits = model(pts, feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / max(1, len(dataloader)):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[Train Teacher] Saved trained teacher model to {save_path}")
    return model
