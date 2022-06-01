import logging
import pathlib
import shutil

import coloredlogs

from modules import global_value as g


def init_logger(
            log_path: pathlib.Path,
            mode='w',
):
    logger = logging.getLogger('')

    stdout_fmt  = '%(asctime)s %(levelname)s: %(message)s'
    coloredlogs.install(level='DEBUG', logger=logger, fmt=stdout_fmt)

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
