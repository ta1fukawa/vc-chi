import pathlib
import sys

import numpy as np

sys.path.append('code/modules/deep-speaker/')

from audio import read_mfcc
from constants import SAMPLE_RATE

wav_dir = pathlib.Path('./dataset/jvs_ver1/wav_all')
dest_dir = pathlib.Path('~/.deep-speaker-wd/my-training/audio-fbanks').expanduser()
dest_dir.mkdir(parents=True, exist_ok=True)

for speaker in sorted(wav_dir.iterdir()):
    if not speaker.is_dir():
        continue

    for speech in sorted(speaker.iterdir()):
        mfcc = read_mfcc(speech, SAMPLE_RATE)  # MFCCと書いてあるが、実際はメルスペクトログラムに変換される(NMELS=64)
        speech_name = speech.stem.split('_')[1]
        dest_path = dest_dir / f'{speaker.name}_{speech_name}.npy'
        np.save(dest_path, mfcc)
