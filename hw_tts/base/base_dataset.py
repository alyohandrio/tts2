from torch.utils.data import Dataset
import os
import librosa
import torch

class BaseDataset(Dataset):
    def __init__(self, path, **batch):
        super().__init__()
        self.files = list(filter(lambda x: x[-4:] == ".wav", os.listdir(path)))
        self.dir = path
    
    def __getitem__(self, i):
        y, sr = librosa.load(os.path.join(self.dir, self.files[i]))
        return torch.tensor(y)
    
    def __len__(self):
        return len(self.files)

