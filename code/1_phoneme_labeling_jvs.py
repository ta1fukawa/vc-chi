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

    wav_dir = pathlib.Path(g.bak_dir)
    lab_dir = pathlib.Path(g.lab_dir)

    lab_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = pathlib.Path(tmp_dir.name)
    tmp_wav_path = tmp_path / 'tmp.wav'

    with open(g.kana_path, 'r') as f:
        kana_list = f.read().splitlines()

    for i, kana in enumerate(kana_list):
        with (tmp_path / f'VOICEACTRESS100_{i + 1:03d}.txt').open('w') as f:
            f.write(kana)

    for wav_speaker in sorted(wav_dir.iterdir()):

        lab_speaker = lab_dir / wav_speaker.name
        lab_speaker.mkdir(parents=True, exist_ok=True)

        for wav_speech in sorted(wav_speaker.iterdir()):
            if (lab_speaker / f'{wav_speech.stem}.lab').exists():
                continue

            print(f'Process: {wav_speaker.name}/{wav_speech.name}\033[K\033[G', end='')

            wave, sr = librosa.load(wav_speech, sr=16000)
            sf.write(tmp_wav_path, wave, sr, subtype='PCM_16')

            args = {
                'wav_file': tmp_wav_path,
                'input_yomi_file': tmp_path / f'{wav_speech.stem}.txt',
                'output_seg_file': lab_speaker / f'{wav_speech.stem}.lab',
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
