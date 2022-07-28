import pathlib
import sys

import numpy as np
import multiprocessing

sys.path.append('code/modules/deep-speaker/')

from audio import read_mfcc
from constants import SAMPLE_RATE

wav_dir = pathlib.Path('./dataset/podcast/ja/wav24k_seg')
dest_dir = pathlib.Path('~/.deep-speaker-wd/my-training/audio-fbanks').expanduser()
dest_dir.mkdir(parents=True, exist_ok=True)

def process(speaker):
    speeches = sorted(speaker.iterdir())
    if len(speeches) <= 10:
        return

    for speech in speeches:
        mfcc = read_mfcc(speech, SAMPLE_RATE)  # MFCCと書いてあるが、実際はメルスペクトログラムに変換される(NMELS=64)
        speech_name = speech.stem.replace('_', '-').replace('podcast', '')
        speaker_name = f'podcast-{speaker.name[:16]}-{speaker.name[32+1+16:]}'.replace('_', '-')
        dest_path = dest_dir / f'{speaker_name}_{speech_name}.npy'
        print(dest_path)
        np.save(dest_path, mfcc)

pool = multiprocessing.Pool(processes=32)
pool.map(process, sorted(wav_dir.iterdir()))
