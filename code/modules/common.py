import datetime
import logging
import pathlib
import shutil

import coloredlogs
import torch
import yaml

from modules import global_value as g


def init_logger(
    log_path: pathlib.Path,
    mode='w',
):
    logger = logging.getLogger('')

    stdout_fmt  = '%(asctime)s %(levelname)s: %(message)s'
    coloredlogs.install(level='INFO', logger=logger, fmt=stdout_fmt)

    logflie_fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    handler = logging.FileHandler(log_path, mode=mode)
    handler.setFormatter(logging.Formatter(logflie_fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


def backup_codes(
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    ext_list: list = ['.py', '.sh', '.yaml', '.json'],
):
    dst_dir.mkdir(parents=True)
    for src_path in src_dir.glob('**/*'):
        if src_path.is_dir():
            continue
        if src_path.suffix in ext_list:
            dst_path = dst_dir / src_path.relative_to(src_dir)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)


def custom_init(
    config_path: pathlib.Path,
    code_id: str,
    run_id_format: str,
):
    g.code_id = code_id
    g.run_id  = datetime.datetime.now().strftime(run_id_format)

    g.work_dir = pathlib.Path('wd', g.code_id, g.run_id)
    g.work_dir.mkdir(parents=True)

    init_logger(g.work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    backup_codes(pathlib.Path(__file__).parent, g.work_dir / 'code')

    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.debug(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)
    for k, v in config[g.vocoder].items():
        setattr(g, k, v)

    if torch.cuda.is_available():
        g.device = torch.device('cuda')
    else:
        g.device = torch.device('cpu')
