
import sys

import matplotlib

matplotlib.use('Agg')
import librosa
import matplotlib.pyplot as plt
import numpy as np
import parallel_wavegan.utils
import scipy.signal
import soundfile as sf
import torch
import torchaudio
# import wavenet_vocoder
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler

# from tqdm import tqdm

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
    
    wave = wave / np.max(np.abs(wave))

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

    fil = scipy.signal.firwin(255, norm_cutoff, pass_zero=False)
    wave = scipy.signal.lfilter(fil, 1, wave)

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


def save_mel_img(mel, path, vmin=-10, vmax=2):
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.imshow(mel.T, cmap='magma', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.savefig(path)
    plt.close()


g._mel_basis = None
def wave2mel(wave):
    if g._mel_basis is None:
        g._mel_basis = librosa.filters.mel(
            sr=g.sample_rate,
            n_fft=g.fft_size,
            n_mels=g.num_mels,
            fmin=g.fmin,
            fmax=g.fmax
        )

    spec = librosa.stft(
        wave,
        n_fft=g.fft_size,
        hop_length=g.hop_size,
        win_length=g.win_length,
        window=g.window,
        pad_mode='reflect'
    )
    mel = np.dot(g._mel_basis, np.abs(spec)).T
    if g.vocoder == 'melgan':
        mel = np.log10(np.clip(mel, a_min=1e-5, a_max=None))
    elif g.vocoder == 'waveglow':
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    return mel.astype(np.float32)


g._waveglow_model = None
def mel2wave_waveglow(mel):
    if g._waveglow_model is None:
        g._waveglow_model = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_vocoder().to(g.device)

    mel = torch.from_numpy(mel.T).unsqueeze(0).to(g.device)

    with torch.no_grad():
        wave, _ = g._waveglow_model(mel, mel.shape)

    wave = wave.view(-1).cpu().numpy()
    wave = wave / np.max(np.abs(wave))

    return wave


g._wavegan_model = None
def mel2wave_melgan(mel):
    if g._wavegan_model is None:
        g._wavegan_model = parallel_wavegan.utils.load_model(g.wavegan_model_path)
        g._wavegan_model.remove_weight_norm()
        g._wavegan_model.eval().to(g.device)

    scaler = StandardScaler()
    scaler.mean_  = parallel_wavegan.utils.read_hdf5(g.wavegan_stats_path, 'mean')
    scaler.scale_ = parallel_wavegan.utils.read_hdf5(g.wavegan_stats_path, 'scale')
    scaler.n_features_in_ = scaler.mean_.shape[0]
    mel = scaler.transform(mel)

    with torch.no_grad():
        wave = g._wavegan_model.inference(mel)

    wave = wave.view(-1).cpu().numpy()
    wave = wave / np.max(np.abs(wave))

    return wave
