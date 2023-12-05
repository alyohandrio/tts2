from hw_tts.base import BaseDiscriminator
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class MPDHead(nn.Module):
    def __init__(self, p, alpha):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(1 if i == 1 else 2 ** (5 + i - 1), 2 ** (5 + i), (5, 1), stride=(3, 1), padding=(2, 0))) for i in range(1, 5)])
        self.conv1 = weight_norm(nn.Conv2d(2 ** (5 + 4), 1024, (5, 1), padding='same'))
        self.conv2 = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding='same'))

    def forward(self, x):
        fms = []
        if x.shape[-1] % self.p != 0:
            x = F.pad(x, (0, self.p - x.shape[-1] % self.p))
        x = x.view(x.shape[0], 1, x.shape[-1] // self.p, self.p)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.alpha)
            fms += [x]
        x = F.leaky_relu(self.conv1(x), self.alpha)
        fms += [x]
        x = self.conv2(x)
        fms += [x]
        return x, fms

class MPD(nn.Module):
    def __init__(self, alpha, ps=[2, 3, 5, 7, 11]):
        super().__init__()
        self.layers = nn.ModuleList([MPDHead(ps[i], alpha) for i in range(len(ps))])

    def forward(self, audio, generated_audio, detach=False, **batch):
        if detach:
            generated_audio = generated_audio.detach()
        xs = []
        fms = []
        xs_real = []
        fms_real = []
        for layer in self.layers:
            cur_x, cur_fms = layer(generated_audio)
            xs += [cur_x]
            fms += [cur_fms]
            cur_x, cur_fms = layer(audio)
            xs_real += [cur_x]
            fms_real += [cur_fms]
        return xs, xs_real, fms, fms_real
