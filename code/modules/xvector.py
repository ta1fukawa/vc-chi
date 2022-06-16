import argparse
import datetime
import logging
import pathlib
import traceback

import matplotlib; matplotlib.use('Agg')
import torch
import torch.utils.tensorboard
import torchinfo
import yaml

from modules import global_value as g
from modules import common
from modules import dataset
from modules import model
from modules import audio
from modules import vgg_perceptual_loss
from modules import ssim_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=pathlib.Path, default='config.yml')
    parser.add_argument('--gpu',         type=int, default=0)

    try:
        main(**vars(parser.parse_args()))
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
    finally:
        logging.info('Done')
        logging.shutdown()

        torch.cuda.empty_cache()
