import argparse
import logging
import pathlib
import sys
import tempfile
import traceback

import librosa
import soundfile as sf

sys.path.append('code/julius4seg')

from sample.run_segment import run_segment

from julius4seg.sp_inserter import ModelType
from modules import common
from modules import global_value as g


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    wav_dir = pathlib.Path(g.wav_dir)
    lab_dir = pathlib.Path(g.lab_dir)

    lab_dir.mkdir(parents=True, exist_ok=True)

    with open(g.kana_path, 'r') as f:
        yomi_list = f.read().splitlines()

    for i, wav in enumerate(sorted(sorted(wav_dir.iterdir())[0].iterdir())):
        logging.info(f'Process: {wav.name}')

        wave, sr = librosa.load(wav, sr=16000)
        wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
        wav_path = pathlib.Path(wav_file.name)
        sf.write(wav_path, wave, sr, subtype='PCM_16')

        with tempfile.NamedTemporaryFile('w', suffix='.txt') as f:
            f.write(yomi_list[i])
            f.seek(0)

            args = {
                'wav_file': wav_path,
                'input_yomi_file': pathlib.Path(f.name),
                'output_seg_file': lab_dir / f'{wav.stem}.lab',
                'input_yomi_type': 'katakana',
                'like_openjtalk': False,
                'input_text_file': None,
                'output_text_file': None,
                'hmm_model': g.hmm_path,
                'model_type': ModelType.gmm,
                'padding_second': 0,
                'options': None
            }

            try:
                run_segment(**args, only_2nd_path=False)
            except:
                run_segment(**args, only_2nd_path=True)

            wav_file.close()

    yomi_dir.cleanup()


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
