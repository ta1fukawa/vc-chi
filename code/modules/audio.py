
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

    if g.highpass_cutoff is not None:
        wave = np.clip(wave, -1., 1.)
        wave = _low_cut_filter(wave, g.highpass_cutoff)

    wave = wave / np.max(np.abs(wave))

    wave = np.pad(wave, (0, g.fft_size), mode='constant')
    mel  = wave2mel(wave)

    wave_len = mel.shape[0] * g.hop_size
    wave = wave[:wave_len]

    return wave, mel


def save_wav_file(wave, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, g.sample_rate)


def save_wave_img(wave, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if type(wave) == torch.Tensor:
        wave = wave.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.ylim(-1, 1)
    plt.plot(wave, color='magenta', linewidth=0.5, alpha=0.5)
    plt.savefig(path)
    plt.close()


def _low_cut_filter(wave, cutoff):
    nyquist = g.sample_rate // 2
    norm_cutoff = cutoff / nyquist

    fil = scipy.signal.firwin(255, norm_cutoff, pass_zero=False)
    wave = scipy.signal.lfilter(fil, 1, wave)

    return wave


def load_mel_data(path):
    if path.suffix == '.npy':
        mel = mel.numpy()
    elif path.suffix == '.pt':
        mel = mel.to(g.device)
    else:
        raise ValueError('Unsupported mel format')

    return mel


def save_mel_data(mel, path):
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

    if type(mel) == torch.Tensor:
        mel = mel.detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.imshow(mel.T, cmap='magma', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.savefig(path)
    plt.close()


def save(name, mel=None, wave=None, vmin=-10, vmax=2, ext='pt', image_ext='png'):
    assert mel is not None or wave is not None

    if mel is None:
        mel = wave2mel(wave)
    if type(mel) == torch.Tensor:
        mel = mel.detach().cpu().numpy()

    mel_data_path = g.work_dir / 'mel_data' / f'{name}.{ext}'
    mel_data_path.parent.mkdir(parents=True, exist_ok=True)
    save_mel_data(mel, mel_data_path)

    mel_img_path = g.work_dir / 'mel_img' / f'{name}.{image_ext}'
    mel_img_path.parent.mkdir(parents=True, exist_ok=True)
    save_mel_img(mel, mel_img_path, vmin=vmin, vmax=vmax)

    if wave is None:
        if g.vocoder == 'melgan':
            wave = mel2wave_melgan(mel)
        elif g.vocoder == 'waveglow':
            wave = mel2wave_waveglow(mel)

    wav_file_path = g.work_dir / 'wav' / f'{name}.wav'
    wav_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav_file(wave, wav_file_path)

    wave_img_path = g.work_dir / 'wave_img' / f'{name}.{image_ext}'
    wave_img_path.parent.mkdir(parents=True, exist_ok=True)
    save_wave_img(wave, wave_img_path)


g._mel_basis = None
def gen_mel_basis():
    if g._mel_basis is None:
        g._mel_basis = librosa.filters.mel(
            sr=g.sample_rate,
            n_fft=g.fft_size,
            n_mels=g.num_mels,
            fmin=g.fmin,
            fmax=g.fmax
        )


def wave2mel(wave):
    spec = wave2spec(wave)
    mel  = spec2mel(spec)

    return mel


def wave2spec(wave):
    spec = librosa.stft(
        wave,
        n_fft=g.fft_size,
        hop_length=g.hop_size,
        win_length=g.win_length,
        window=g.window,
        pad_mode='constant'
    )
    spec = np.abs(spec)

    return spec.astype(np.float32)


def spec2wave(spec):
    wave = librosa.istft(
        spec,
        hop_length=g.hop_size,
        win_length=g.win_length,
        window=g.window,
        length=len(spec) * g.hop_size
    )

    return wave


def spec2mel(spec):
    gen_mel_basis()

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


g._stft_window = None
def fast_stft(wave):
    if g._stft_window is None:
        g._stft_window = librosa.filters.get_window('hann', g.win_length, fftbins=True).reshape((-1, 1))
        g._fft = librosa.core.fft.get_fftlib()
        g._n_columns = 63

    y_frames = librosa.util.frame(wave, frame_length=g.fft_size, hop_length=g.hop_size)
    stft_matrix = np.empty((int(1 + g.fft_size // 2), y_frames.shape[1]), dtype=np.float32, order='F')

    for bl_s in range(0, y_frames.shape[1], g._n_columns):
        stft_matrix[:, bl_s:bl_s + g._n_columns] = np.abs(g._fft.rfft(
            g._stft_window * y_frames[:, bl_s:bl_s + g._n_columns], axis=0
        )).astype(np.float32)

    return stft_matrix
