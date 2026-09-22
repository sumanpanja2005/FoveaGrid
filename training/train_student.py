import os
import torch
from torch.utils.data import DataLoader
from models.teacher import TeacherSegmentationModel
from models.student import StudentSegmentationModel
from models.distillation import DistillationLoss
from training.dataset import LidarScanDataset

def train_student(
    data_dir: str,
    teacher_path: str = "models/teacher.pt",
    student_save_path: str = "models/student.pt",
    epochs: int = 2,
    lr: float = 0.001
):
    os.makedirs(os.path.dirname(student_save_path), exist_ok=True)
    dataset = LidarScanDataset(data_dir)

    teacher = TeacherSegmentationModel()
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
    teacher.eval()

    student = StudentSegmentationModel()

    if len(dataset) == 0:
        print(f"[Train Student] Warning: No dataset scans found in {data_dir}. Creating initialized student weights at {student_save_path}.")
        torch.save(student.state_dict(), student_save_path)
        return student

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    distill_criterion = DistillationLoss(alpha=0.6, beta=0.4, temperature=4.0)

    print(f"[Train Student] Distilling Student Model over {len(dataset)} scans for {epochs} epochs...")
    student.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for sample in dataloader:
            pts = sample["points"][0]
            feats = sample["features"][0]
            labels = sample["labels"][0]

            with torch.no_grad():
                teacher_logits = teacher(pts, feats)

            optimizer.zero_grad()
            student_logits = student(pts, feats)
            loss = distill_criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss / max(1, len(dataloader)):.4f}")

    torch.save(student.state_dict(), student_save_path)
    print(f"[Train Student] Saved distilled student model to {student_save_path}")
    return student
