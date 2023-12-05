from hw_tts.base import BaseLoss
import torch

class GeneratorLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        loss = 0
        for x in xs:
            cur_loss = torch.mean((x - 1) ** 2)
            loss = loss + cur_loss
        return loss
