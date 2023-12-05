from hw_tts.base import BaseDiscriminator
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class MSDHead(nn.Module):
    def __init__(self, alpha, use_spectral_norm=False):
        super().__init__()
        self.alpha = alpha
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv1 = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fms = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.alpha)
            fms += [x]
        x = self.conv1(x)
        fms += [x]
        return x, fms

class MSD(BaseDiscriminator):
    def __init__(self, alpha):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MSDHead(alpha, use_spectral_norm=True),
            MSDHead(alpha),
            MSDHead(alpha),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, audio, generated_audio, detach=False, **batch):
        x = generated_audio
        x_real = audio
        if detach:
            x = x.detach()
        xs = []
        fms = []
        xs_real = []
        fms_real = []
        for i, layer in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
                x_real = self.meanpools[i-1](x_real)
            cur_x, cur_fms = layer(x)
            xs += [cur_x]
            fms += [cur_fms]
            cur_x, cur_fms = layer(x_real)
            xs_real += [cur_x]
            fms_real += [cur_fms]
        return xs, xs_real, fms, fms_real
