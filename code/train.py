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

from modules import common
from modules import global_value as g
from modules import model


def main():
    # おまじない
    assert torch.cuda.is_available()  # GPUが使えるか確認

    # 実行時変数
    g.code_id = 'vc-chi-a'
    g.run_id  = datetime.datetime.now().strftime('%Y%m%d/%H%M%S')

    # 超定数
    g.device = torch.device('cuda:0')

    # ファイル出力先設定
    work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    work_dir.mkdir(parents=True)

    # ロガー初期化
    common.init_logger(work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    # コードバックアップ
    common.backup_codes(pathlib.Path(__file__).parent, work_dir / 'code')

    ### 準備 ###

    logging.info('Prepare')

    # モデル定義
    net = model.Net()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = []

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