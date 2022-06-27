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

    predict()


def predict():
    emb_dir = pathlib.Path(g.emb_dir)

    embs = []
    for emb_file in sorted(emb_dir.iterdir()):
        emb = torch.load(emb_file)
        embs.append(emb)

    cos_sim = []
    for emb_i in embs:
        cos_sim_i = []
        for emb_j in embs:
            cos_sim_ij = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0).item()
            cos_sim_i.append(cos_sim_ij)
        cos_sim.append(cos_sim_i)

    with open(g.work_dir / 'emb_cossim.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cos_sim)

    vec_distance = []
    for emb_i in embs:
        vec_distance_i = []
        for emb_j in embs:
            vec_distance_ij = torch.norm(emb_i - emb_j, p=2).item()
            vec_distance_i.append(vec_distance_ij)
        vec_distance.append(vec_distance_i)

    with open(g.work_dir / 'emb_dffdis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(vec_distance)


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
