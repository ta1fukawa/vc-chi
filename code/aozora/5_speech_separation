import pathlib
import sys
import warnings

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)

import inaSpeechSegmenter
import numpy as np
import soundfile as sf
import tensorflow as tf
import torchaudio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

src_dir = pathlib.Path('dataset/aozora/wav')
dst_dir = pathlib.Path('dataset/aozora/wav_seg')

pad_size = 4000

seg_model = inaSpeechSegmenter.Segmenter(vad_engine='sm', detect_gender=False, batch_size=1024)

i = int(sys.argv[1])
for wav_path in sorted(src_dir.glob('*/*'))[i * 100:(i + 1) *100]:
    print(wav_path, end='')
    speaker_path = wav_path.parent
    (dst_dir / speaker_path.stem).mkdir(parents=True, exist_ok=True)

    data = seg_model(wav_path)

    wave, sr = torchaudio.load(wav_path)
    wave /= wave.max()
    wave = wave.squeeze().numpy()

    seg_id = 1
    for state, start, end in data:
        if state == 'speech' and 16 >= end - start >= 3:
            seg_name = f'{wav_path.stem}_{seg_id:03d}.wav'
            sf.write(dst_dir / speaker_path.stem / seg_name, np.pad(wave[int(start * sr):int(end * sr)], (pad_size, pad_size), 'constant'), sr)
            seg_id += 1

    print(wav_path, seg_id)
