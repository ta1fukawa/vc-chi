import argparse
import csv
import logging
import pathlib
import sys
import traceback

import torch

sys.path.append('code')

from modules import audio, common
from modules import global_value as g
from modules import xvector


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

    wav_dir = pathlib.Path(g.bak_dir)

    embs = []

    for speaker in sorted(wav_dir.iterdir()):
        if not speaker.is_dir():
            continue

        speaker_mels = []

        for wav in sorted(speaker.iterdir()):
            print(f'Process: {speaker.name}/{wav.name}\033[K\033[G', end='')

            wave, mel = audio.load_wav(wav)

            speaker_mels.append(padding(torch.from_numpy(mel)))

        if len(speaker_mels) > 0:
            speaker_mels = torch.stack(speaker_mels, dim=0).to(g.device)
            with torch.no_grad():
                _, emb = net(speaker_mels)
            emb = torch.mean(emb, dim=0).cpu()
            embs.append(emb)

    print('\033[K\033[G', end='')

    cos_sim = []
    for emb_i in embs:
        cos_sim_i = []
        for emb_j in embs:
            cos_sim_ij = torch.mean(torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0)).item()
            cos_sim_i.append(cos_sim_ij)
        cos_sim.append(cos_sim_i)

    with open(g.work_dir / 'cos_sim_mat.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cos_sim)


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
