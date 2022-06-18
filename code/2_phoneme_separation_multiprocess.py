import argparse
import csv
import logging
import pathlib
import traceback
import multiprocessing

import librosa
import numpy as np
import pyworld

from modules import audio, common
from modules import global_value as g


def main(config_path):
    common.custom_init(config_path, '%Y%m%d/%H%M%S')

    wav_dir = pathlib.Path(g.wav_dir)
    lab_dir = pathlib.Path(g.lab_dir)
    pnm_mel_dir = pathlib.Path(g.pnm_mel_dir)
    pnm_spc_dir = pathlib.Path(g.pnm_spc_dir)

    pnm_mel_dir.mkdir(parents=True, exist_ok=True)
    pnm_spc_dir.mkdir(parents=True, exist_ok=True)

    pool = multiprocessing.Pool(processes=g.processer_num)
    pool.map(process, sorted(wav_dir.iterdir()))


def process(speaker):
    wav_dir = pathlib.Path(g.wav_dir)
    lab_dir = pathlib.Path(g.lab_dir)
    pnm_mel_dir = pathlib.Path(g.pnm_mel_dir)
    pnm_spc_dir = pathlib.Path(g.pnm_spc_dir)

    speaker_pnm_spc = []
    speaker_pnm_mel = []

    logging.info(f'Process: {speaker.name}')

    for wav in sorted(speaker.iterdir()):
        if not wav.is_file() or wav.suffix != '.wav':
            continue

        wave, sr = librosa.load(wav, sr=g.sample_rate, dtype=np.float64, mono=True)

        lab_path = lab_dir / f'{wav.stem}.lab'
        with lab_path.open('r') as f:
            reader = csv.reader(f, delimiter='\t')
            labels = [row for row in reader]

        if g.extract_envelope:
            _, sp, _ = extract_acoustic_features(wave, sr)
            sp = sp.astype(np.float32)

            separation_rate = 200

            for start_sec, end_sec, phoneme in labels:
                if phoneme in ['silB', 'silE', 'sp']:
                    continue

                start_frame = max(int(float(start_sec) * separation_rate), 0)
                end_frame   = min(int(float(end_sec)   * separation_rate), len(sp))

                spc = sp[start_frame:end_frame].T
                mel = audio.spec2mel(spc)

                speaker_pnm_spc.append(spc.T)
                speaker_pnm_mel.append(mel)
        else:
            for start_sec, end_sec, phoneme in labels:
                if phoneme in ['silB', 'silE', 'sp']:
                    continue

                start_sample = int(float(start_sec) * sr)
                end_sample   = int(float(end_sec)   * sr)

                spc = audio.wave2spec(wave[start_sample:end_sample])
                mel = audio.spec2mel(spc)

                speaker_pnm_spc.append(spc.T)
                speaker_pnm_mel.append(mel)

    np.save(pnm_spc_dir / f'{speaker.name}.npy', speaker_pnm_spc, allow_pickle=True)
    np.save(pnm_mel_dir / f'{speaker.name}.npy', speaker_pnm_mel, allow_pickle=True)


def extract_acoustic_features(wave, sr, mode='dio'):
    if mode == 'dio':
        _f0, t = pyworld.dio(wave, sr) # 基本周波数の抽出
    elif mode == 'harvest':
        _f0, t = pyworld.harvest(wave, sr)
    f0 = pyworld.stonemask(wave, _f0, t, sr) # 洗練させるらしい f0 (n, )
    sp = pyworld.cheaptrick(wave, f0, t, sr) # スペクトル包絡の抽出 spectrogram (n, f)
    ap = pyworld.d4c(wave, f0, t, sr) # 非周期性指標の抽出 aperiodicity (n, f)

    return f0, sp, ap

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
