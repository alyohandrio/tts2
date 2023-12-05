from abc import abstractmethod
from typing import Union

import numpy as np
import torch.nn as nn
from torch import Tensor


class BaseGenerator(nn.Module):
    """
    Base class for all generators
    """

    def __init__(self, **batch):
        super().__init__()

    @abstractmethod
    def forward(self, **batch) -> Union[Tensor, dict]:
        """
        Forward pass logic.
        Can return a torch.Tensor (it will be interpreted as waveform) or a dict.

        :return: Model output
        """
        raise NotImplementedError()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
