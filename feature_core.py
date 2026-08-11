from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

def extract(path: Path, sr=16000, n_mels=128, n_frames=128):
    x, actual = sf.read(path, dtype='float32')
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1: x = x.mean(1)
    if actual != sr: x = signal.resample_poly(x, sr, actual)
    x = x / (np.max(np.abs(x)) + 1e-8)
    # Bound the analysis window to keep extraction deterministic and fast; segmentation already supplies the cough event.
    max_samples = int(sr * 4.0)
    if len(x) > max_samples: x = x[:max_samples]
    _, _, z = signal.stft(x, fs=sr, nperseg=512, noverlap=384, nfft=512, boundary='zeros')
    power = np.abs(z) ** 2
    hz = np.linspace(0, sr / 2, power.shape[0])
    edges = np.linspace(0, sr / 2, n_mels + 2)
    mel = np.zeros((n_mels, power.shape[1]))
    for i in range(n_mels):
        left, center, right = edges[i:i + 3]
        weights = np.maximum(0, np.minimum((hz - left) / (center - left), (right - hz) / (right - center)))
        mel[i] = weights @ power
    log_mel = np.log(np.maximum(mel, 1e-10))
    if log_mel.shape[1] < n_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])), mode='constant', constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :n_frames]
    rms = np.sqrt(np.mean(x * x))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    centroid = float(np.sum(hz[:, None] * power) / (np.sum(power) + 1e-9))
    bandwidth = float(np.sqrt(np.sum(((hz[:, None] - centroid) ** 2) * power) / (np.sum(power) + 1e-9)))
    complementary = np.array([rms, zcr, centroid, bandwidth, len(x) / sr, np.max(np.abs(x)), np.mean(log_mel), np.std(log_mel)], dtype='float32')
    return log_mel.astype('float32'), complementary
