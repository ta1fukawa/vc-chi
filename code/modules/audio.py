
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
    if path.suffix == '.npy':
        mel = mel.numpy()
    elif path.suffix == '.pt':
        mel = mel.to(g.device)
    else:
        raise ValueError('Unsupported mel format')

    return mel


def save_mel(mel, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == '.npy':
        if type(mel) == torch.Tensor:
            mel = mel.detach().cpu().numpy()
        np.save(path, mel)
    elif path.suffix == '.pt':
        if type(mel) == np.ndarray:
            mel = torch.from_numpy(mel)
        torch.save(mel, path)
    else:
        raise ValueError('Unsupported mel format')


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


g._waveglow_model = None
def mel2wave_waveglow(mel):
    if g._waveglow_model is None:
        g._waveglow_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        g._waveglow_model = g._waveglow_model.remove_weightnorm(g._waveglow_model)
        g._waveglow_model.eval().to(g.device)

    mel = torch.from_numpy(mel.T).unsqueeze(0).to(g.device)

    with torch.no_grad():
        wave = g._waveglow_model.infer(mel)
    
    wave = wave.squeeze(0).cpu().numpy()

    return wave


g._melgan_model = None
def mel2wave_melgan(mel):
    if g._melgan_model is None:
        from parallel_wavegan.models.melgan import MelGANGenerator
        from parallel_wavegan.layers.pqmf import PQMF

        g._melgan_model = MelGANGenerator(
            in_channels=g.num_mels,
            out_channels=4,
            kernel_size=7,
            channels=384,
            upsample_scales=[5, 5, 3],
            stack_kernel_size=3,
            stacks=4,
            use_weight_norm=True,
            use_causal_conv=False
        )
        g._melgan_model.load_state_dict(torch.load('./model/train_nodev_jsut_multi_band_melgan.v2/checkpoint-1000000steps.pkl')["model"]["generator"])
        g._melgan_model.register_stats('./model/train_nodev_jsut_multi_band_melgan.v2/stats.h5')
        g._melgan_model.pqmf = PQMF(subbands=4)
        g._melgan_model.remove_weight_norm()
        g._melgan_model = g._melgan_model.eval().to(g.device)

    mel = torch.from_numpy(mel).to(g.device)

    with torch.no_grad():
        wave = g._melgan_model.inference(mel)

    wave = wave.squeeze(1).cpu().numpy()

    return wave

