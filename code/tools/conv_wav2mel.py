import argparse
import datetime
import logging
import os
import pathlib
import sys

import torch
import torchaudio
import torchinfo
import yaml

sys.path.append('code')

from modules import common, dataset
from modules import global_value as g
from modules import model
from modules import audio


def main(config_path, encoder_path, gpu=0):
    g.code_id = '_wav2mel'
    g.run_id  = datetime.datetime.now().strftime('%Y%m%d/%H%M%S')

    work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    work_dir.mkdir(parents=True)

    common.init_logger(work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')

    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.info(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)

    if gpu >= 0:
        assert torch.cuda.is_available()
        g.device = torch.device(f'cuda:{gpu}')
    else:
        g.device = torch.device('cpu')
    
    emb_encoder = torch.load(encoder_path, map_location=g.device).eval()
    torchinfo.summary(emb_encoder, input_size=(100, 128, 80), col_names=['output_size', 'num_params'])

    wav_dir = pathlib.Path(g.wav_dir)
    mel_dir = pathlib.Path(g.mel_dir)
    agl_dir = pathlib.Path(g.agl_dir)
    emb_dir = pathlib.Path(g.emb_dir)

    mel_dir.mkdir(parents=True, exist_ok=True)
    agl_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        (mel_dir / speaker.name).mkdir(parents=True, exist_ok=True)
        (agl_dir / speaker.name).mkdir(parents=True, exist_ok=True)

        speaker_mels = []

        for wav in sorted(speaker.iterdir()):
            if not wav.is_file() or wav.suffix != '.wav':
                continue

            print(f'\rProcess: {speaker.name}/{wav.name}', end='')

            wave, sr = torchaudio.load(str(wav))
            wave, sr = audio.norm_wave(wave, **config['norm_wave'])
            mel, agl = audio.wave2mel(wave, **config['mel_spec'])

            torch.save(mel, str(mel_dir / speaker.name / f'{wav.stem}.pt'))
            torch.save(agl, str(agl_dir / speaker.name / f'{wav.stem}.pt'))

            speaker_mels.append(mel)

        if len(speaker_mels) > 0:
            emb = audio.mel2embed(speaker_mels, emb_encoder, **config['mel2embed'])
            torch.save(emb, str(emb_dir / f'{speaker.name}.pt'))


def pad_seq(mel, seg_len):
    if len(mel) < seg_len:
        len_pad = seg_len - len(mel)
        mel = torch.cat((mel, torch.zeros(len_pad, mel.shape[1])), dim=0)
    else:
        mel = mel[:seg_len]
    return mel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',  type=pathlib.Path, default='audio_config.yml')
    parser.add_argument('--encoder_path', type=pathlib.Path, default='model/dvector.pt')
    parser.add_argument('--gpu',          type=int, default=0)

    main(**vars(parser.parse_args()))
