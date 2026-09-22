import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train_teacher import train_teacher
from training.train_student import train_student

def main():
    parser = argparse.ArgumentParser(description="FoveaGrid Teacher & Student Model Trainer")
    parser.add_argument("--data_dir", type=str, default="data/synthetic", help="Synthetic dataset directory")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    print("[Training] Phase 1: Training Teacher Model...")
    train_teacher(data_dir=args.data_dir, save_path="models/teacher.pt", epochs=args.epochs, lr=args.lr)

    print("[Training] Phase 2: Distilling Student Model...")
    train_student(data_dir=args.data_dir, teacher_path="models/teacher.pt", student_save_path="models/student.pt", epochs=args.epochs, lr=args.lr)

    print("[Training] Completed training pipeline!")

if __name__ == "__main__":
    main()
