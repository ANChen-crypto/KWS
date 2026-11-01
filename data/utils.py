import torch
import torchaudio
import librosa

import numpy as np
from scipy.signal import lfilter, butter
import pywt


def fix_length(wav, target_len):
    if len(wav) > target_len:
        start = np.random.randint(0, len(wav) - target_len + 1)
        wav = wav[start:start + target_len]
    else:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav

def random_time_shift(x, right_bound_s=0.1, left_bound_s=0.1, shift_max=0.2, sr=16000):
    max_shift = int(shift_max * sr)

    energy = short_time_energy(x)
    max_frame = np.argmax(energy)
    center = (max_frame * 160)

    left_bound = int(left_bound_s * sr)
    right_bound = int(right_bound_s * sr)
    left_limit  = left_bound
    right_limit = len(x) - right_bound
    shift_min = max(-max_shift, left_limit - center)
    shift_max = min(max_shift, right_limit - center)

    if shift_min >= shift_max:
        shift = 0
    else:
        shift = np.random.randint(shift_min, shift_max)

    return np.roll(x, shift)

def agc_process(x, target_rms=0.15, eps=1e-8):
    rms = np.sqrt(np.mean(x**2))
    scale = target_rms / (rms + eps)
    return x * scale

def volume_perturb(x, gain=[0.5, 1.5]):
    g = np.random.uniform(gain[0], gain[1])
    x_aug = x * g
    return x_aug

def random_time_stretch(wav, rate_range=(0.9, 1.1)):
    wav = torch.from_numpy(wav)
    rate = np.random.uniform(*rate_range)
    stretch = torchaudio.transforms.TimeStretch()
    spec = torch.stft(wav, n_fft=400, return_complex=True)
    spec = stretch(spec, rate)
    wav = torch.istft(spec, n_fft=400)
    return wav.numpy()

def add_noise(x, snr_db):
    x_power = np.mean(x**2)
    noise_power = x_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), x.shape)
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, -1.0, 1.0)
    return x_noisy

def augment_wav(wav, sr, target_len):
    original_wav = wav
    wav = random_time_shift(wav, shift_max=0.8, sr=sr)
    wav = volume_perturb(wav, gain=[0.2, 4])
    # wav = add_noise(wav, snr_db=20)
    if torch.rand(1).item() > 0.7:
        wav = pre_emphasis(wav)
    # wav = random_time_stretch(wav, rate_range=(0.9, 1.1))
    # wav = fix_length(wav, target_len=target_len)
    return wav

def wavlet_denoise(wav, level=1):
    coeffs = pywt.wavedec(wav, "db4", level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(wav)))

    new_coeffs = []
    new_coeffs.append(coeffs[0])
    for i in range(1, len(coeffs)):
        scale = np.sqrt(2 * np.log(len(coeffs[i])))
        new_coeffs.append(
            pywt.threshold(coeffs[i], sigma * scale, mode="soft")
        )
    
    return pywt.waverec(new_coeffs, "db4")[:len(wav)]

def bandpass(x, fs=16000, low=80, high=4000, order=8):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return lfilter(b, a, x)

def pre_emphasis(x, coef=0.95):
    return np.append(x[0], x[1:] - coef * x[:-1])

def short_time_energy(y, frame_length=400, hop_length=160):
    frames = librosa.util.frame(
        y, frame_length=frame_length, hop_length=hop_length
    )
    energy = np.sum(frames ** 2, axis=0)
    return energy


if __name__ == "__main__":
    import librosa
    import sounddevice as sd
    path = "./ken/seg/ym1_1_seg01.wav"
    # path = "C:/Users/asdfa/OneDrive/Desktop/kws server/sound.wav"
    y, _ = librosa.load(path, sr=16000)
    sd.play(y, 16000)
    sd.wait()
    # clean_y = wavlet_denoise(y, level=2)
    clean_y = pre_emphasis(y)
    sd.play(clean_y, 16000)
    sd.wait()
