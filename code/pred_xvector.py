import argparse
import logging
import pathlib
import traceback

import numpy as np
import torch

from modules import audio, common, dataset
from modules import global_value as g
from modules import xvector, ssim_loss, vgg_perceptual_loss



def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    net = xvector.Net().to(g.device)
    logging.debug(f'MODEL: {net}')

    if g.model_load_path is not None:
        net.load_state_dict(torch.load(g.model_load_path, map_location=g.device))
        logging.debug(f'LOAD MODEL: {g.model_load_path}')

    predict(net)


def predict(net):
    net.eval()

    wav_dir = pathlib.Path(g.wav_dir)
    mel_dir = pathlib.Path(g.mel_dir)
    emb_dir = pathlib.Path(g.emb_dir)

    mel_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        (mel_dir / speaker.name).mkdir(parents=True, exist_ok=True)

        speaker_mels = []

        for wav in sorted(speaker.iterdir()):
            if not wav.is_file() or wav.suffix != '.wav':
                continue

            print(f'Process: {speaker.name}/{wav.name}\033[K\033[G', end='')

            wave, mel = audio.load_wav(wav)
            audio.save_mel_data(mel, mel_dir / speaker.name / f'{wav.stem}.pt')

            speaker_mels.append(padding(torch.from_numpy(mel)))

        if len(speaker_mels) > 0:
            speaker_mels = torch.stack(speaker_mels, dim=0).to(g.device)
            with torch.no_grad():
                _, emb = net(speaker_mels)
            emb = torch.mean(emb, dim=0).cpu()
            torch.save(emb, str(emb_dir / f'{speaker.name}.pt'))

    print('Finished\033[K\033[G')


def padding(data):
    if len(data) < g.seg_len:
        len_pad = g.seg_len - len(data)
        data = torch.cat((data, torch.zeros(len_pad, data.shape[1])), dim=0)
    else:
        data = data[:g.seg_len]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='xvector_config.yml')

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
