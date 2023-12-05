from hw_tts.base import BaseGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, alpha):
        super().__init__()
        self.alpha = alpha
        self.layers = nn.ModuleList()
        for m in range(len(dilations)):
            self.layers.append(nn.ModuleList())
            for ell in range(len(dilations[m])):
                self.layers[-1].append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilations[m][ell], padding='same')))

    def forward(self, x):
        for block in self.layers:
            out = x
            for layer in block:
                out = layer(F.leaky_relu(out, self.alpha))
            x = out + x
        return x

class MRF(nn.Module):
    def __init__(self, channels, kr, Dr, alpha):
        super().__init__()
        self.alpha = alpha
        self.blocks = nn.ModuleList([ResBlock(channels, kr[i], Dr[i], alpha) for i in range(len(kr))])

    def forward(self, x):
        out = None
        for block in self.blocks:
            if out == None:
                out = block(x)
            else:
                out = block(x) + out
        return out

class Generator(BaseGenerator):
    def __init__(self, ku, hu, kr, Dr, alpha):
        super().__init__()
        self.alpha = alpha
        self.pre_conv = weight_norm(nn.Conv1d(80, hu, 7, padding='same'))
        self.mrfs = nn.ModuleList([MRF(hu // (2 ** (i + 1)), kr, Dr, alpha) for i in range(len(ku))])
        self.convs = nn.ModuleList([weight_norm(nn.ConvTranspose1d(hu // (2 ** i), hu // (2 ** (i + 1)), ku[i], stride=ku[i] // 2, padding=ku[i] // 4)) for i in range(len(ku))])
        self.post_conv = weight_norm(nn.Conv1d(hu // (2 ** len(ku)), 1, 7, padding='same'))

    def forward(self, spectrogram, **batch):
        x = self.pre_conv(spectrogram)
        for i in range(len(self.convs)):
            x = self.convs[i](F.leaky_relu(x, self.alpha))
            x = self.mrfs[i](x)
        x = self.post_conv(F.leaky_relu(x, self.alpha))
        x = F.tanh(x)
        return x
