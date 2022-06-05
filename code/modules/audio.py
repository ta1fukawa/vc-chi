import sys

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.append('code')

from modules import global_value as g


def save_spec_fig(c, t, r, q):
    c = c.squeeze(0).detach().cpu().numpy()
    t = t.squeeze(0).detach().cpu().numpy()
    r = r.squeeze(0).detach().cpu().numpy()
    q = q.squeeze(0).detach().cpu().numpy()

    (g.work_dir / 'spec').mkdir(parents=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(c.T[::-1], cmap='magma', vmin=g.spec_fig['min_db'], vmax=g.spec_fig['max_db'])
    plt.savefig(str(g.work_dir / 'spec' / 'source_content.png'))

    plt.figure(figsize=(10, 6))
    plt.imshow(t.T[::-1], cmap='magma', vmin=g.spec_fig['min_db'], vmax=g.spec_fig['max_db'])
    plt.savefig(str(g.work_dir / 'spec' / 'target_content.png'))

    plt.figure(figsize=(10, 6))
    plt.imshow(r.T[::-1], cmap='magma', vmin=g.spec_fig['min_db'], vmax=g.spec_fig['max_db'])
    plt.savefig(str(g.work_dir / 'spec' / 'predict_before.png'))

    plt.figure(figsize=(10, 6))
    plt.imshow(q.T[::-1], cmap='magma', vmin=g.spec_fig['min_db'], vmax=g.spec_fig['max_db'])
    plt.savefig(str(g.work_dir / 'spec' / 'predict_after.png'))


def save_mel_wave(c, t, r, q, angle):
    source_wave  = mel2wave(c[0], angle[0], ver=2, **g.mel_spec)
    target_wave  = mel2wave(t[0], angle[0], ver=2, **g.mel_spec)
    predict_wave = mel2wave(q[0], angle[0], ver=2, **g.mel_spec)

    source_wave_  = source_wave.squeeze(0).detach().cpu().numpy()
    target_wave_  = target_wave.squeeze(0).detach().cpu().numpy()
    predict_wave_ = predict_wave.squeeze(0).detach().cpu().numpy()

    (g.work_dir / 'wav').mkdir(parents=True)

    sf.write(str(g.work_dir / 'wav' / f'source.wav'),  source_wave_,  g.mel_spec['sample_rate'])
    sf.write(str(g.work_dir / 'wav' / f'target.wav'),  target_wave_,  g.mel_spec['sample_rate'])
    sf.write(str(g.work_dir / 'wav' / f'predict.wav'), predict_wave_, g.mel_spec['sample_rate'])


def wave2mel(wave, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms / 1000),
        power=None,
    ).to(g.device)(wave)
    radii = torch.abs(spec)
    angle = torch.angle(spec)

    mel = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        n_stft=n_fft // 2 + 1,
    ).to(g.device)(radii)

    mel = 20 * torch.log10(torch.clamp(mel, min=1e-9))
    mel = (mel - ref_db) / (dc_db - ref_db)

    mel   = mel.squeeze(0).T
    angle = angle.squeeze(0).T
    return mel, angle


def mel2wave(mel, angle, ver=1, **kwargs):
    if ver == 1:
        wave = mel2wave_v1(mel, angle, **kwargs)
    elif ver == 2:
        wave = mel2wave_v2(mel, **kwargs)
    return wave


def mel2wave_v1(mel, angle, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
    mel   = mel.T.unsqueeze(0)
    angle = angle.T.unsqueeze(0)

    mel = mel * (dc_db - ref_db) + ref_db
    mel = 10**(mel / 20)

    radii = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
    ).to(g.device)(mel)

    spec = radii * torch.exp(1j * angle[:, :, :radii.size(2)])

    wave = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms / 1000),
    ).to(g.device)(spec)

    return wave


def mel2wave_v2(mel, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
    if not hasattr(g, 'waveglow'):
        # g.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
        # g.waveglow = g.waveglow.remove_weightnorm(g.waveglow)
        # g.waveglow = g.waveglow.to('cuda')
        # g.waveglow.eval()
        g.waveglow = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_vocoder().to(g.device)

    mel = mel.T.unsqueeze(0)

    with torch.no_grad():
        # wave = g.waveglow.infer(mel)
        wave, _ = g.waveglow(mel, mel.shape)

    return wave


def mel2embed(mels, encoder, seg_len):
    mels = torch.stack([
        mel[:seg_len] for mel in mels if len(mel) >= seg_len
    ], dim=0).to(g.device)

    with torch.no_grad():
        embeds = encoder(mels)

    embed = torch.mean(embeds, dim=0)
    return embed
