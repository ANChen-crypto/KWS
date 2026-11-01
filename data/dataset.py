import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from pathlib import Path
import os
import yaml
from tqdm import tqdm
import librosa
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import soundfile as sf

from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import hyp1f1, gamma

from data import augment_wav, agc_process, pre_emphasis
from modeling import PhonemeLM


class_map = {
    "1_1": 0,
    "1_2": 1,
    "1_3": 2,
    "1_4": 3,
    "1_5": 3,
    "1_6": 3,
    "2_1": 0,
    "2_2": 1,
    "2_3": 2,
    "2_4": 3,
    "3_1": 0,
    "3_2": 1,
    "3_3": 2,
    "3_4": 2,
    "3_5": 3,
    "9_9": 4,
    "noise": 4,
    "unk": 4,
}

class_map_13 = {
    "1_1": 1,
    "1_2": 10,
    "1_3": 4,
    "1_4": 7,
    "1_5": 7,
    "1_6": 7,
    "2_1": 2,
    "2_2": 11,
    "2_3": 5,
    "2_4": 8,
    "3_1": 3,
    "3_2": 12,
    "3_3": 6,
    "3_4": 6,
    "3_5": 9,
    "unk": 0,
    "noise": 0,
    "9_9": 0,
}

phoneme_map = {
    "1_1": ["k", "iu", "m", "ia", "o", "h"],                 # 救命喔
    "1_2": ["g", "ua", "p", "uah", "t", "o", "a"],           # 我跌倒了
    "1_3": ["s", "u", "iau"],                                # 需要
    "1_4": ["b", "o", "s", "u", "iau"],                      # 不需要
    "1_5": ["b", "o", "l", "a"],                             # 沒啦
    "1_6": ["m", "b", "ian"],                                # 毋免

    "2_1": ["j", "iu", "m", "ing", "j", "iu", "m", "ing"],   # 救命救命
    "2_2": ["w", "o", "d", "ie", "d", "ao", "l", "e"],       # 我跌倒了
    "2_3": ["x", "u", "y", "ao", "b", "ang", "j", "u"],      # 需要幫助
    "2_4": ["b", "u", "x", "u", "y", "ao"],                  # 不需要

    "3_1": ["h", "eh", "l", "p", "h", "eh", "l", "p"],       # help help
    "3_2": ["ay", "f", "eh", "l"],                           # I fell
    "3_3": ["n", "iy", "d", "h", "eh", "l", "p"],            # need help
    "3_4": ["y", "eh", "s"],                                 # yes
    "3_5": ["n", "ow", "n", "iy", "d"],                      # no need

    "9_9": ["unk"],
    "unk": ["unk"],
    "noise": ["sil"],
}

sequences = [k for k in phoneme_map.values()]
phonemelm = PhonemeLM(sequences=sequences)

phoneme_vocab = {
    'a': 1,
    'ang': 2,
    'ao': 3,
    'ay': 4,
    'b': 5,
    'd': 6,
    'e': 7,
    'eh': 8,
    'f': 9,
    'g': 10,
    'h': 11,
    'ia': 12,
    'ian': 13,
    'iau': 14,
    'ie': 15,
    'ing': 16,
    'iu': 17,
    'iy': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'ow': 25,
    'p': 26,
    's': 27,
    'sil': 28,
    't': 29,
    'u': 30,
    'ua': 31,
    'uah': 32,
    'unk': 33,
    'w': 34,
    'x': 35,
    'y': 36,

    '<bos>': 0,
    '<pad>': 37,
    '<eos>': 38,
}
phoneme_vocab_reverse = {v: k for k, v in phoneme_vocab.items()}

phonemes_to_keyword = {tuple(v): class_map[k] for k, v in phoneme_map.items()}

keyword_ids2class = {}
for k, phonemes in phoneme_map.items():
    ids = [phoneme_vocab[p] for p in phonemes]
    keyword_ids2class[tuple(ids)] = class_map[k]

yaml_path = "data/dataset.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class CTCMelSpecDataset(Dataset):
    def __init__(self, train_mode, num_samples=None):
        super().__init__()

        self.train_mode = train_mode

        self.audio_length = config["audio_length"]
        self.sampling_rate = config["sampling_rate"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]

        self.phonemes = set()
        for seq in phoneme_map.values():
            self.phonemes.update(seq)
        self.phonemes = sorted(list(self.phonemes))
        self.phoneme_vocab = {p: i+1 for i, p in enumerate(sorted(self.phonemes))}
        print("phoneme vocab", self.phoneme_vocab)
        self.blank = 0

        label_list = []
        path_list = []

        name_list = "name_list_train" if train_mode else "name_list_test"
        patterns = [".mp3", ".wav", ".m4a"]
        speaker_counter = defaultdict(int)
        cnt = 0
        for pattern in patterns:
            for name in config[name_list]:
                root = Path(f"D:/keywords/{name}")
                paths = [
                    str(p) for p in root.rglob(f"*{pattern}")
                    # if len(p.relative_to(root).parts) == 2
                ]
                random.shuffle(paths)
                with tqdm(paths) as pbar:
                    for i, path in enumerate(pbar):
                        if i % 1000 == 0:
                            pbar.set_postfix_str(str(path))
                        basename = os.path.basename(path)
                        
                        if "noise_train" in path or "noise_test" in path:
                            class_id = "noise"
                            label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                        elif "unk" in path:
                            class_id = "unk"
                            label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                        else:
                            if train_mode:
                                splits = path.split("\\")
                                speaker_id = f"{splits[-3]}_{splits[-2]}_{basename[2:5]}"
                                if not (path.split("\\")[-2] == "seg" or path.split("\\")[-2] == "seg_denoised"):
                                    continue
                                if speaker_counter[speaker_id] > num_samples:
                                    continue
                                if "1_9" in path or "2_9" in path or "2_5" in path or "9_9"in path:
                                    continue
                                speaker_counter[speaker_id] += 1
                                class_id = basename[2:5]
                                label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                            else:
                                if not (path.split("\\")[-2] == "seg" or path.split("\\")[-2] == "seg_denoised"):
                                    continue
                                class_id = basename[2:5]
                                label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                                
                        label_list.append(label)
                        path_list.append(path)

        self.label_list = label_list
        self.path_list = path_list
        
    def __len__(self):
        return len(self.path_list)

    def collator(self, bx):
        mels, labels, paths, audios, target_lengths = zip(*bx)

        targets = torch.cat(
            [torch.tensor(l, dtype=torch.long) for l in labels]
        )
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long
        )

        return {
            "x": torch.stack(mels),
            "path": list(paths),
            "audio": audios,
            "targets": targets,
            "target_lengths": target_lengths,
            "labels": labels,
        }
    
    def __getitem__(self, index):
        path = self.path_list[index]
        label = self.label_list[index]
        audio, _ = librosa.load(self.path_list[index], sr=self.sampling_rate)
        if len(audio) > self.audio_length:
            start = np.random.randint(0, len(audio) - self.audio_length)
            audio = audio[start:start + self.audio_length]
        else:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))
        if self.train_mode and label != class_map["noise"] and label != class_map["unk"]:
            audio = augment_wav(audio, sr=self.sampling_rate, target_len=self.audio_length)
        else:
            audio = pre_emphasis(agc_process(audio))
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=False,
        )
        log_mel = librosa.power_to_db(mel_spec)

        log_mel = torch.from_numpy(log_mel).float()
        return log_mel, label, path, audio, len(label)

class CTCMelSpecNpzDataset(Dataset):
    def __init__(self, train_mode):
        super().__init__()

        self.train_mode = train_mode

        self.sampling_rate = config["sampling_rate"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]

        self.phonemes = set()
        for seq in phoneme_map.values():
            self.phonemes.update(seq)
        self.phonemes = sorted(list(self.phonemes))
        self.phoneme_vocab = {p: i+1 for i, p in enumerate(sorted(self.phonemes))}

        label_list = []
        path_list = []

        self.bos = phoneme_vocab["<bos>"]
        self.eos = phoneme_vocab["<eos>"]
        self.pad = phoneme_vocab["<pad>"]

        class_counter = Counter()

        if train_mode:
            root1 = Path(f"G:/npzs_stride_0.01s_train")
            root2 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root2.iterdir()
                if p.is_dir() and not (p.name.endswith("person_2_xinyi")\
                 or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
            paths = list(root1.rglob(f"*.npz")) + wav_paths
        else:
            root1 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root1.iterdir()
                if p.is_dir() and (p.name.endswith("person_2_xinyi")\
                 or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
        skip = 0
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                if train_mode:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6", "9_9"]:
                        skip += 1
                        continue
                    class_id = basename[2:5]
                    label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                else:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6", "9_9"]:
                        skip += 1
                        continue
                    class_id = basename[2:5]
                    label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                label_list.append(label)
                path_list.append(str(path))

                class_counter[class_id] += 1
        
        # only load unk in train_mode, not in test mode
        if train_mode:
            root = Path(f"G:/npz_unk_train")
        else:
            root = Path(f"G:/npz_unk_test")

        paths = root.rglob(f"*.npz")
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i > 3500000:
                    break
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                class_id = "unk"
                label = [self.phoneme_vocab[p] for p in phoneme_map[class_id]]
                label_list.append(label)
                path_list.append(str(path))

                class_counter[class_id] += 1

        self.label_list = label_list
        self.path_list = path_list
        self.len = len(self.path_list)

        if True:
            print("\n****************************************")
            print("class distribution in training set")
            for k, v in class_counter.items():
                print(f"class {k}: {v}")
            print("\n****************************************")
        
    def __len__(self):
        return len(self.path_list)

    def collator(self, bx):
        mels, labels, paths, target_lengths, importance = zip(*bx)

        targets = torch.cat(
            [torch.tensor(l, dtype=torch.long) for l in labels]
        )
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long
        )
        importance = torch.tensor(
            importance, dtype=torch.float32
        )

        # decoder_labels = [
        #     torch.tensor([self.bos] + label + [self.eos], dtype=torch.long)
        #     for label in labels
        # ]

        # decoder_labels = pad_sequence(
        #     decoder_labels,
        #     batch_first=True,
        #     padding_value=self.pad
        # )
        # decoder_input = decoder_labels[:, :-1]
        # decoder_target = decoder_labels[:, 1:]
        
        return {
            "x": torch.stack(mels),
            "path": list(paths),
            "targets": targets,
            "target_lengths": target_lengths,
            "labels": labels,
            "importance": importance,

            # "decoder_input": decoder_input,
            # "decoder_target": decoder_target,
        }

    def apply_gain(self, M, vmin, vmax):
        gain = random.uniform(vmin, vmax)
        gain_db = 20 * np.log10(gain)
        return M + gain_db
    
    def __getitem__(self, index):
        path = self.path_list[index]
        label = self.label_list[index]

        if Path(path).suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = "mel" if "mel" in data.files else "mels"
                log_mel = data[key][:35].copy()

        if self.train_mode:
            if "testing_set_wav" in path:
                log_mel = self.apply_gain(log_mel, vmin=0.5, vmax=1.1)
                
            f = random.randint(0, 5)
            f0 = random.randint(0, self.n_mels - f)
            log_mel[f0:f0+f, :] = 0

            t = random.randint(0, 15)
            t0 = random.randint(0, 197 - t)

            log_mel[:, t0:t0+t] = 0

        log_mel = torch.from_numpy(log_mel).float()

        # name = os.path.basename(path)
        # if "scale" in name:
        #     scale = float(name.split("_scale")[1].split("_")[0])
        #     if scale < 0.5:
        #         importance = 2.0
        #     else:
        #         importance = 1.0
        # else:
        #     importance = 1.0
        # if (len(label) == 1):
        #     importance = 1.0
        # else:
        #     importance = 4.0
        if ("testing_set_wav" in path):
            importance = 3.0
        else:
            importance = 1.0

        return log_mel, label, path, len(label), importance

class SSLMelSpecNpzDataset(Dataset):
    def __init__(self, train_mode):
        super().__init__()

        self.train_mode = train_mode

        self.sampling_rate = config["sampling_rate"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]

        path_list = []

        class_counter = Counter()

        if train_mode:
            root1 = Path(f"G:/npzs_stride_0.01s_train")
            root2 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root2.iterdir()
                if p.is_dir() and not (p.name.endswith("person_2_xinyi")\
                 or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
            paths = list(root1.rglob(f"*.npz")) + wav_paths
        else:
            root1 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root1.iterdir()
                if p.is_dir() and (p.name.endswith("person_2_xinyi")\
                 or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
        skip = 0
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                if train_mode:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6", "9_9"]:
                        skip += 1
                        continue
                else:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6", "9_9"]:
                        skip += 1
                        continue
                path_list.append(str(path))

        if train_mode:
            root = Path(f"G:/npz_unk_train")
        else:
            root = Path(f"G:/npz_unk_test")

        paths = root.rglob(f"*.npz")
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i > 3500000:
                    break
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                path_list.append(str(path))

        self.path_list = path_list
        self.len = len(self.path_list)

        if True:
            print("\n****************************************")
            print("class distribution in training set")
            for k, v in class_counter.items():
                print(f"class {k}: {v}")
            print("\n****************************************")
        
    def __len__(self):
        return len(self.path_list)

    def collator(self, bx):
        cleans, noisys, importance = zip(*bx)

        importance = torch.tensor(
            importance, dtype=torch.float32
        )
        
        return {
            "x": torch.stack(noisys),
            "targets": torch.stack(cleans),
            "importance": importance,
        }

    def apply_gain(self, M, vmin, vmax):
        gain = random.uniform(vmin, vmax)
        gain_db = 20 * np.log10(gain)
        return M + gain_db
    
    def apply_mask(self, clean):
        noisy = clean.copy()

        f = random.randint(0, 5)
        f0 = random.randint(0, self.n_mels - f)
        noisy[f0:f0+f, :] = 0

        t = random.randint(0, 20)
        t0 = random.randint(0, 197 - t)

        noisy[:, t0:t0+t] = 0
        return noisy
    
    def __getitem__(self, index):
        path = self.path_list[index]
        
        if Path(path).suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = "mel" if "mel" in data.files else "mels"
                log_mel = data[key][:35].copy()

        if self.train_mode:
            if "testing_set_wav" in path:
                log_mel = self.apply_gain(log_mel, vmin=0.3, vmax=1.1)
        
        noisy_log_mel = self.apply_mask(log_mel)
        noisy_log_mel = torch.from_numpy(noisy_log_mel).float()
        log_mel = torch.from_numpy(log_mel).float()

        if ("testing_set_wav" in path):
            importance = 3.0
        else:
            importance = 1.0

        return log_mel, noisy_log_mel, importance
    
class MelSpecDataset(Dataset):
    def __init__(self, train_mode):
        super().__init__()

        self.train_mode = train_mode
        
        self.sampling_rate = config["sampling_rate"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]

        label_list = []
        path_list = []

        class_counter = Counter()

        if train_mode:
            root1 = Path(f"G:/npzs_stride_0.01s_train")
            root2 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root2.iterdir()
                if p.is_dir() and not (p.name.endswith("person_2_xinyi")\
                    or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
            paths = list(root1.rglob(f"*.npz")) + wav_paths
        else:
            root1 = Path(f"G:/testing_set_wav/npz_o")
            folders = [
                p for p in root1.iterdir()
                if p.is_dir() and (p.name.endswith("person_2_xinyi")\
                    or p.name.endswith("an_xinyi"))
            ]
            wav_paths = []
            for folder in folders:
                wav_paths.extend(folder.rglob("*.npz"))
            paths = wav_paths
        skip = 0
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                if train_mode:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6"]:
                        skip += 1
                        continue
                else:
                    if not basename[2:5] in ["1_1", "1_2", "1_3", "1_6"]:
                        skip += 1
                        continue

                class_id = basename[2:5]
                label = class_map[class_id]
                    
                label_list.append(label)
                path_list.append(str(path))

                class_counter[class_id] += 1
        
        # only load unk in train_mode, not in test mode
        if train_mode:
            root = Path(f"G:/npz_unk_train")
        else:
            root = Path(f"G:/npz_unk_test")

        paths = root.rglob(f"*.npz")
        with tqdm(paths, disable=True) as pbar:
            for i, path in enumerate(pbar):
                if i % 100000 == 0:
                    print(f"{i}: {path}", flush=True)
                basename = os.path.basename(path)
                
                class_id = "unk"
                label = class_map[class_id]
                label_list.append(label)
                path_list.append(str(path))

                class_counter[class_id] += 1

        self.label_list = label_list
        self.path_list = path_list
        self.len = len(self.path_list)

        if True:
            print("\n****************************************")
            print("class distribution in training set")
            for k, v in class_counter.items():
                print(f"class {k}: {v}")
            print("\n****************************************")
        
        
    def __len__(self):
        return self.len

    def collator(self, bx):
        mels, labels, paths, importance = zip(*bx)
        importance = torch.tensor(
            importance, dtype=torch.float32
        )

        return {
            "x": torch.stack(mels),
            "classes": torch.tensor(labels, dtype=torch.long),
            "path": list(paths),
            "importance": importance
        }

    def apply_gain(self, M, vmin, vmax):
        gain = random.uniform(vmin, vmax)
        gain_db = 20 * np.log10(gain)
        return M + gain_db
    
    def __getitem__(self, index):
        path = self.path_list[index]
        label = self.label_list[index]

        try:
            log_mel = np.load(path)["mel"]
        except:
            log_mel = np.load(path)["mels"]     # (F, T)

        log_mel = log_mel[:35]
        if "testing_set_wav" in path:
            log_mel = self.apply_gain(log_mel, vmin=0.5, vmax=1.1)

        if self.train_mode:
            f = random.randint(0, 5)
            f0 = random.randint(0, self.n_mels - f)
            log_mel[f0:f0+f, :] = 0

            t = random.randint(0, 15)
            t0 = random.randint(0, 197 - t)

            log_mel[:, t0:t0+t] = 0

        log_mel = torch.from_numpy(log_mel).float()

        if ("testing_set_wav" in path):
            importance = 4.0
        else:
            importance = 1.0
        return log_mel, label, path, importance

class CTCMelSpecNoisyNpzDataset(Dataset):
    def __init__(self, wav_path):
        super().__init__()

        self.audio_length = config["audio_length"]
        self.sampling_rate = config["sampling_rate"]
        self.n_fft = config["n_fft"]
        self.n_mels = config["n_mels"]
        self.hop_length = config["hop_length"]

        self.phonemes = set()
        for seq in phoneme_map.values():
            self.phonemes.update(seq)
        self.phonemes = sorted(list(self.phonemes))
        self.phoneme_vocab = {p: i+1 for i, p in enumerate(sorted(self.phonemes))}
        
        self.bos = phoneme_vocab["<bos>"]
        self.eos = phoneme_vocab["<eos>"]
        self.pad = phoneme_vocab["<pad>"]

        noise_suppress_db = 0
        self.wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        self.segs = sliding_window_view(self.wav, (int(self.sampling_rate * 2)))[::int(self.sampling_rate * 0.2)]
        self.fp = "hsiu_noisy_saved/"

        self.window = np.sqrt(np.hanning(self.n_fft + 1)[:-1])
        self.n_bins = self.n_fft // 2 + 1
        self.hop_size = 320

        self.noise = np.ones(self.n_bins, dtype=np.float64)
        self.old_ps = np.ones(self.n_bins, dtype=np.float64)
        self.S = np.zeros(self.n_bins, dtype=np.float64)
        self.Smin = np.full(self.n_bins, np.inf)
        self.Stmp = np.full(self.n_bins, np.inf)
        self.update_prob = np.ones(self.n_bins, dtype=bool)

        self.prior = np.ones(self.n_bins, dtype=np.float64)
        self.post = np.ones(self.n_bins, dtype=np.float64)
        self.zeta = np.ones(self.n_bins, dtype=np.float64)
        self.gain = np.ones(self.n_bins, dtype=np.float64)
        self.gain2 = np.ones(self.n_bins, dtype=np.float64)
        self.gain_floor = np.ones(self.n_bins, dtype=np.float64)

        self.noise = np.ones(self.n_bins, dtype=np.float64)
        self.n_adapt = 0
        self.min_count = 0

        self.beta = 0.03
        self.beta_1 = 1.0 - self.beta

        self.gain_floor_value = 10.0 ** (
            noise_suppress_db / 20.0
        )

    def __len__(self):
        return len(self.segs)

    def collator(self, bx):
        mel = bx

        return {
            "x": torch.stack(mel),
        }

    @staticmethod
    def hypergeom_gain(theta):
        """
        Ephraim-Malah loudness-domain 增益函數：
        Gamma(1.25)^2 * 1F1(-0.25, 1, -theta) / sqrt(theta)
        """
        theta = np.maximum(theta, 1e-8)

        result = (
            gamma(1.25) ** 2
            * hyp1f1(-0.25, 1.0, -theta)
            / np.sqrt(theta)
        )

        return np.nan_to_num(
            result, nan=1.0, posinf=1.0, neginf=0.0
        )

    def update_noise_prob(self, ps):
        """
        對應 SpeexDSP update_noise_prob() 的主要邏輯。
        ps 是目前各頻率 bin 的功率。
        """

        self.S[1:-1] = (
            0.8 * self.S[1:-1]
            + 0.05 * ps[:-2]
            + 0.10 * ps[1:-1]
            + 0.05 * ps[2:]
        )

        self.S[0] = 0.8 * self.S[0] + 0.2 * ps[0]
        self.S[-1] = 0.8 * self.S[-1] + 0.2 * ps[-1]

        if self.n_adapt == 1:
            self.Smin.fill(0.0)
            self.Stmp.fill(0.0)

        if self.n_adapt < 100:
            min_range = 15
        elif self.n_adapt < 1000:
            min_range = 50
        elif self.n_adapt < 10000:
            min_range = 150
        else:
            min_range = 300

        if self.min_count > min_range:
            self.min_count = 0
            self.Smin = np.minimum(self.Stmp, self.S)
            self.Stmp = self.S.copy()
        else:
            self.Smin = np.minimum(self.Smin, self.S)
            self.Stmp = np.minimum(self.Stmp, self.S)

        # SpeexDSP 判斷：
        # 0.4 * S[i] > Smin[i]
        self.update_prob = 0.4 * self.S > self.Smin

    def process_frame(self, frame):
        x = np.zeros(self.n_fft, dtype=np.float32)
        x[:len(frame)] = frame

        x_fft = np.fft.rfft(x * self.window)
        ps = np.abs(x_fft) ** 2

        self.n_adapt += 1
        self.min_count += 1

        if self.n_adapt == 1:
            self.old_ps = ps.copy()
        
        self.update_noise_prob(ps)

        update_mask = ((~self.update_prob) | (ps < self.noise))

        self.noise[update_mask] = np.maximum(0.0, self.beta_1 * self.noise[update_mask] + self.beta * ps[update_mask])

        post_snr = ps / np.maximum(self.noise, 1e-12)
        self.post = np.maximum(post_snr, 1e-8)

        prior_from_previous = self.old_ps / np.maximum(self.noise, 1e-12)
        self.prior = np.maximum(0.0, 0.98 * prior_from_previous + 0.02 * np.maximum(self.post - 1.0, 0.0))

        self.zeta = 0.7 * self.zeta + 0.3 * self.prior

        avg_snr = np.mean(self.zeta)
        Pframe = 0.1 + 0.899 * (avg_snr / (avg_snr + 1))
        Pframe = np.clip(Pframe, 0.01, 0.999)

        P1 = 0.199 + 0.8 * (self.zeta / (self.zeta + 1))
        q = np.clip(1 - Pframe * P1, 1e-6, 0.999999)

        prior_ratio = self.prior / (1 + self.prior)
        theta = prior_ratio * (1.0 + self.post)

        MM = self.hypergeom_gain(theta)
        self.gain = np.clip(prior_ratio * MM, 0.0, 1.0)

        gain2_probability = 1.0 / (
            1.0
            + (q / np.maximum(1.0 - q, 1e-6))
            * (1.0 + self.prior)
            * np.exp(-np.minimum(theta, 700.0))
        )

        self.gain_floor.fill(self.gain_floor_value)

        self.gain = np.minimum(
            self.gain,
            1.0,
        )

        self.gain = np.maximum(
            self.gain,
            self.gain_floor,
        )

        # gain2 = [p*sqrt(gain) + (1-p)*sqrt(gain_floor)]^2
        p = np.clip(gain2_probability, 0.0, 1.0)

        self.gain2 = (
            p * np.sqrt(self.gain)
            + (1.0 - p) * np.sqrt(self.gain_floor)
        ) ** 2

        self.gain2 = np.clip(self.gain2, 0.0, 1.0)

        Y = x_fft * self.gain2

        self.old_ps = (
            0.2 * self.old_ps
            + 0.8 * (self.gain ** 2) * ps
        )

        y = np.fft.irfft(Y, n=self.n_fft)

        return y * self.window

    def process(self, audio):

        audio = np.asarray(audio, dtype=np.float64).reshape(-1)

        if len(audio) == 0:
            return audio.copy()

        pad = self.n_fft
        padded = np.pad(audio, (pad, pad))

        output = np.zeros_like(padded)
        weight = np.zeros_like(padded)

        for start in range(
            0,
            len(padded) - self.n_fft + 1,
            self.hop_size,
        ):
            frame = padded[start:start + self.n_fft]
            y = self.process_frame(frame)

            output[start:start + self.n_fft] += y
            weight[start:start + self.n_fft] += self.window ** 2

        valid = weight > 1e-10
        output[valid] /= weight[valid]

        result = output[pad:pad + len(audio)]
        return result
    
    def __getitem__(self, index):

        wav = self.segs[index]
        # clean = self.process(wav)

        # if not os.path.exists(self.fp):
        #     os.mkdir(self.fp)
        # sf.write(f"{self.fp}\{index}.wav", wav, self.sampling_rate)

        noisy_mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=40,
            center=False
        )
        noisy_mel = librosa.power_to_db(noisy_mel)

        # clean_mel = librosa.feature.melspectrogram(
        #     y=clean,
        #     sr=self.sampling_rate,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     n_mels=40,
        #     center=False
        # )
        # clean_mel = librosa.power_to_db(clean_mel)

        # plt.subplot(1,2,1)
        # plt.imshow(noisy_mel, origin="lower")
        # plt.subplot(1,2,2)
        # plt.imshow(clean_mel, origin="lower")
        # plt.show()

        # name = os.path.basename(path)
        # if "scale" in name:
        #     scale = float(name.split("_scale")[1].split("_")[0])
        #     if scale < 0.5:
        #         importance = 2.0
        #     else:
        #         importance = 1.0
        # else:
        #     importance = 1.0
        # if (len(label) == 1):
        #     importance = 1.0
        # else:
        #     importance = 4.0

        return torch.tensor(noisy_mel, dtype=torch.float32)[:35]