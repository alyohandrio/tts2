from abc import abstractmethod
from typing import Union

import numpy as np
from torch import Tensor
import torch.nn as nn


class BaseLoss(nn.Module):
    """
    Base class for all losses
    """

    def __init__(self, **batch):
        super().__init__()

    @abstractmethod
    def forward(self, **batch) -> Union[Tensor, dict]:
        """
        Forward pass logic.
        Can return a torch.Tensor (it will be interpreted as loss) or a dict.

        :return: Loss output
        """
        raise NotImplementedError()
