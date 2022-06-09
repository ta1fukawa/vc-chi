
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import wavenet_vocoder
import torch
from tqdm import tqdm

sys.path.append('code')

from modules import global_value as g


def load_wav(path):
    sr, wave = wavfile.read(path)

    if wave.dtype == np.int16:
        wave = wave / 2**15
    elif wave.dtype == np.int32:
        wave = wave / 2**31
    elif wave.dtype == np.float32:
        pass
    else:
        raise ValueError('Unsupported wav format')

    if sr != g.sample_rate:
        wave = librosa.resample(wave, sr, g.sample_rate)

    wave = np.clip(wave, -1., 1.)
    wave = _low_cut_filter(wave, g.highpass_cutoff)

    wave = np.pad(wave, (0, g.fft_size), mode='constant')
    mel  = wave2mel(wave)

    wave_len = mel.shape[0] * g.hop_size
    wave = wave[:wave_len]

    return wave, mel


def save_wav(wave, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, g.sample_rate)


def _low_cut_filter(wave, cutoff):
    nyquist = g.sample_rate // 2
    norm_cutoff = cutoff / nyquist

    from scipy.signal import firwin, lfilter

    fil = firwin(255, norm_cutoff, pass_zero=False)
    wave = lfilter(fil, 1, wave)

    return wave


def load_mel(path):
    mel = np.load(path)

    return mel


def save_mel(mel, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mel)


def save_mel_img(mel, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.imshow(mel.T, cmap='magma', aspect='auto', origin='lower', vmin=-10, vmax=2)
    plt.savefig(path)
    plt.close()


g._mel_basis = None
def wave2mel(wave):
    if g._mel_basis is None:
        g._mel_basis = librosa.filters.mel(g.sample_rate, g.fft_size, g.num_mels, g.fmin, g.fmax)

    spec = librosa.stft(wave, n_fft=g.fft_size, hop_length=g.hop_size, win_length=g.win_length, window=g.window, pad_mode='constant')
    mel = np.dot(g._mel_basis, np.abs(spec))
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))

    return mel.astype(np.float32).T


g._wavenet_model = None
def mel2wave(mel):
    if g._wavenet_model is None:
        g._wavenet_model = wavenet_vocoder.WaveNet(**g.wavenet_model, scalar_input=True).to(g.device)

        checkpoint = torch.load('model/20180510_mixture_lj_checkpoint_step000320000_ema.pth')
        g._wavenet_model.load_state_dict(checkpoint['state_dict'])

        g._wavenet_model.eval()
        g._wavenet_model.make_generation_fast_()

    wave_length = mel.shape[0] * g.hop_size

    initial_input = torch.zeros(1, 1, 1).fill_(0.0).to(g.device)
    mel = torch.from_numpy(mel.T).unsqueeze(0).to(g.device)

    with torch.no_grad():
        wave = g._wavenet_model.incremental_forward(
            initial_input, c=mel, g=None, T=wave_length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=g.log_scale_min,
        )
    
    wave = wave.view(-1).cpu().numpy()

    return wave
