import torch
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import Dataset

import glob
import os
import yaml

import librosa
import numpy as np


yaml_path = "data/dataset.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class RawAudioDataset(Dataset):
    def __init__(self, load_pretrained_processor=True):
        super().__init__()

        self.audio_length = config["audio_length"]
        self.sampling_rate = config["sampling_rate"]

        audio_list = []
        label_list=  []
        for name in config["name_list"]:
            for path in glob.glob(f"data/{name}/**/*.wav"):
                basename = os.path.basename(path)
                if "9" in basename:
                    continue
                y, _ = librosa.load(path, sr=None)
                label_list.append(int(basename.split("_")[1]) - 1)
                audio_list.append(y)

        self.label_list = np.array(label_list, dtype=np.long)
        self.audio_list = np.array(audio_list)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")\
              if load_pretrained_processor else None

    def __len__(self):
        return len(self.audio_list)
    
    def collater(self, bx: tuple):
        collated_audio = torch.zeros((len(bx), self.audio_length))
        collated_label = torch.zeros((len(bx),), dtype=torch.long)
        for i in range(len(bx)):
            audio, label = bx[i]
            if self.processor:
                collated_audio[i] = torch.from_numpy(
                    self.processor(audio, sampling_rate=self.sampling_rate)["input_values"]
                )
            else:
                pass
            collated_label[i] = torch.tensor([label], dtype=torch.long)
        return {"x": collated_audio, "classes": collated_label}
    
    def __getitem__(self, index):
        return self.audio_list[index], self.label_list[index]
