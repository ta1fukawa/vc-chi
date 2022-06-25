import csv
import datetime
import fcntl
import inspect
import logging
import os
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
    ext_list: list = ['.py', '.sh', '.yml', '.yaml', '.json'],
):
    dst_dir.mkdir(parents=True)
    for src_path in src_dir.glob('**/*'):
        if src_path.is_dir():
            continue
        if src_path.suffix in ext_list:
            dst_path = dst_dir / src_path.relative_to(src_dir)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_path, dst_path)


def torch_reset_seed(
    seed: int = 0,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def custom_init(
    config_path: pathlib.Path,
    run_id_format: str,
    note: str = None,
):
    g.code_id = os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
    g.run_id  = datetime.datetime.now().strftime(run_id_format)

    g.work_dir = pathlib.Path('dest', g.code_id, g.run_id)
    g.work_dir.mkdir(parents=True)

    init_logger(g.work_dir / 'run.log')
    logging.info(f'CODE/RUN: {g.code_id}/{g.run_id}')

    backup_codes(pathlib.Path('code'), g.work_dir / 'code')
    shutil.copy(config_path, g.work_dir / 'config.yml')

    config = yaml.load(config_path.open(mode='r'), Loader=yaml.FullLoader)
    logging.info(f'CONFIG: {config}')

    for k, v in config.items():
        setattr(g, k, v)
    if 'vocoder' in config and g.vocoder in config:
        for k, v in config[g.vocoder].items():
            setattr(g, k, v)

    if torch.cuda.is_available():
        g.device = torch.device('cuda')
    else:
        g.device = torch.device('cpu')

    update_note_status('init', note)

    torch_reset_seed(0)

g._note = None
def update_note_status(
    status: str,
    note: str = None,
):
    if note is not None:
        g._note = note
        logging.info(f'NOTE: {note}')

    if g._note is not None:
        note_path = pathlib.Path('status', f'{g.code_id}.csv')
        if not note_path.exists():
            with note_path.open(mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(['code_id', 'run_id', 'note', 'status'])

        with note_path.open(mode='r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            rows = list(csv.reader(f))
            for i, row in enumerate(rows):
                if row[1] == g.run_id:
                    rows[i] = [g.code_id, g.run_id, g._note, status]
                    break
            else:
                rows.append([g.code_id, g.run_id, g._note, status])
            f.seek(0)
            f.truncate(0)
            writer = csv.writer(f)
            writer.writerows(rows)
            fcntl.flock(f, fcntl.LOCK_UN)
