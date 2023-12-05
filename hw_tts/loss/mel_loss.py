from hw_tts.base import BaseLoss
from torch.nn import L1Loss

class MelLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, raw, generated):
        return L1Loss()(raw, generated)
