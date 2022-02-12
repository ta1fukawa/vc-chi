import argparse
import datetime
import gc
import logging
import pathlib
import traceback

import numpy as np
import torch
import torch.utils.tensorboard
import tqdm
import yaml

from modules import common
from modules import global_value as g
from modules import model


def main(config_path):
    assert torch.cuda.is_available()

    g.code_id = 'apple'
    g.run_id  = datetime.datetime.now().strftime('%Y%m/%d/%H%M%S')

    g.device = torch.device('cuda:0')

    work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    work_dir.mkdir(parents=True)

    common.init_logger(work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')
    
    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.info(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)

    net = model.Net().to(g.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=pathlib.Path)

    args = [
        'config.yaml'
    ]

    try:
        main(**vars(parser.parse_args(args)))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        # おまじない
        gc.collect()
        torch.cuda.empty_cache()