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

    # c, t, r, q = (([c, t, r, q] - np.min(t) + 1e-8) / np.max(t) * 255).astype(np.int32)

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
    source_wave  = mel2wave(c, angle, **g.mel_spec)
    target_wave  = mel2wave(t, angle, **g.mel_spec)
    predict_wave = mel2wave(q, angle, **g.mel_spec)

    source_wave_  = source_wave.squeeze(0).detach().cpu().numpy()
    target_wave_  = target_wave.squeeze(0).detach().cpu().numpy()
    predict_wave_ = predict_wave.squeeze(0).detach().cpu().numpy()

    (g.work_dir / 'wav').mkdir(parents=True)

    sf.write(str(g.work_dir / 'wav' / f'source.wav'),  source_wave_,  g.mel_spec['sample_rate'])
    sf.write(str(g.work_dir / 'wav' / f'target.wav'),  target_wave_,  g.mel_spec['sample_rate'])
    sf.write(str(g.work_dir / 'wav' / f'predict.wav'), predict_wave_, g.mel_spec['sample_rate'])


def norm_wave(wave, sample_rate, norm_db, sil_threshold, sil_duration, preemph):
    effects = [
        ['channels', '1'],
        ['rate', f'{sample_rate}'],
        ['norm', f'{norm_db}'],
        [
            'silence',
            '1',
            f'{sil_duration}',
            f'{sil_threshold}%',
            '-1',
            f'{sil_duration}',
            f'{sil_threshold}%',
        ],
    ]

    wave, sample_rate = torchaudio.sox_effects.apply_effects_tensor(wave, sample_rate, effects)
    wave = torch.cat([wave[:, 0].unsqueeze(-1), wave[:, 1:] - preemph * wave[:, :-1]], dim=-1)

    return wave, sample_rate


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


def mel2wave(mel, angle, sample_rate, fft_window_ms, fft_hop_ms, n_fft, f_min, n_mels, ref_db, dc_db):
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

    spec = radii * torch.exp(1j * angle[:, :radii.shape[1]])

    wave = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=int(sample_rate * fft_window_ms / 1000),
        hop_length=int(sample_rate * fft_hop_ms / 1000),
    ).to(g.device)(spec)

    return wave


def mel2embed(mels, encoder, seg_len):
    mels = torch.stack([
        mel[:seg_len] for mel in mels if len(mel) >= seg_len
    ], dim=0).to(g.device)

    with torch.no_grad():
        embeds = encoder(mels)

    embed = torch.mean(embeds, dim=0)
    return embed
