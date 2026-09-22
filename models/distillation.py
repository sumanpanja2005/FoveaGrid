import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Teacher-Student Knowledge Distillation Loss for LiDAR Semantic Segmentation:
    L_total = alpha * CrossEntropy(student_logits, labels) + beta * T^2 * KL_Divergence(softmax(student_logits/T), softmax(teacher_logits/T))
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss(student_logits, labels)

        # Distillation KL loss
        p_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        loss_kl = F.kl_div(p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)

        total_loss = self.alpha * loss_ce + self.beta * loss_kl
        return total_loss
