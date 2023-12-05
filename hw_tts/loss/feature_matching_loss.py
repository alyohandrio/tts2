from hw_tts.base import BaseLoss
import torch

class FMLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, raw, generated):
        loss = 0
        for r_h, g_h in zip(raw, generated):
            for r_l, g_l in zip(r_h, g_h):
                loss = loss + torch.mean(torch.abs(r_l - g_l))
        return loss
