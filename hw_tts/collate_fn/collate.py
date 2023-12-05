from hw_tts.utils import MelSpectrogram, MelSpectrogramConfig
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    audios = pad_sequence(batch, batch_first=True)
    mels = MelSpectrogram(MelSpectrogramConfig())(audios)
    return {"spectrogram": mels, "audio": audios.unsqueeze(1)}
