from hw_tts.base import BaseLoss
import torch

class DiscriminatorLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, raw, generated):
        loss = 0
        for x_r, x_g in zip(raw, generated):
            cur_loss = torch.mean((x_r - 1) ** 2) + torch.mean((x_g - 1) ** 2)
            loss = loss + cur_loss
        return loss
