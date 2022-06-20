# Run only once

import argparse
import logging
import pathlib
import traceback

import torch

from modules import audio, common
from modules import global_value as g


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    wav_dir = pathlib.Path(g.wav_dir)
    mel_dir = pathlib.Path(g.mel_dir)

    mel_dir.mkdir(parents=True, exist_ok=True)

    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        (mel_dir / speaker.name).mkdir(parents=True, exist_ok=True)

        for wav in sorted(speaker.iterdir()):
            print(f'Process: {speaker.name}/{wav.name}\033[K\033[G', end='')

            wave, mel = audio.load_wav(wav)
            audio.save_mel_data(mel, mel_dir / speaker.name / f'{wav.stem}.pt')

    print('\033[K\033[G', end='')


def padding(data):
    if len(data) < g.seg_len:
        len_pad = g.seg_len - len(data)
        data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
    else:
        data = data[:g.seg_len]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='vc_config.yml')

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
