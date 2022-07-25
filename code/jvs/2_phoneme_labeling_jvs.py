import pathlib
import sys
import tempfile

import librosa
import soundfile as sf

sys.path.append('code/modules/julius4seg')

from sample.run_segment import run_segment
from julius4seg.sp_inserter import ModelType


wav_dir = './dataset/jvs_ver1/wav_all'
lab_dir = './dataset/jvs_ver1/lab'
kana_path = './dataset/jvs_hiho/voiceactoress100_spaced_julius.txt'
hmm_path = './dataset/dictation-kit-4.5/model/phone_m/jnas-mono-16mix-gid.binhmm'


wav_dir = pathlib.Path(wav_dir)
lab_dir = pathlib.Path(lab_dir)

lab_dir.mkdir(parents=True, exist_ok=True)
yomi_dir = tempfile.TemporaryDirectory()
yomi_dir_path = pathlib.Path(yomi_dir.name)

with open(kana_path, 'r') as f:
    yomi_list = f.read().splitlines()

for i, kana in enumerate(yomi_list):
    with (yomi_dir_path / f'VOICEACTRESS100_{i + 1:03d}.txt').open('w') as f:
        f.write(kana)

for wav_speaker in sorted(wav_dir.iterdir()):
    print(f'Processing {wav_speaker.name}')

    lab_speaker = lab_dir / wav_speaker.name
    lab_speaker.mkdir(parents=True, exist_ok=True)

    for wav_speech in sorted(wav_speaker.iterdir()):
        if (lab_speaker / f'{wav_speech.stem}.lab').exists():
            continue

        wave, sr = librosa.load(wav_speech, sr=16000)
        wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
        wav_path = pathlib.Path(wav_file.name)
        sf.write(wav_path, wave, sr, subtype='PCM_16')

        args = {
            'wav_file': wav_path,
            'input_yomi_file': yomi_dir_path / f'{wav_speech.stem}.txt',
            'output_seg_file': lab_speaker / f'{wav_speech.stem}.lab',
            'input_yomi_type': 'katakana',
            'like_openjtalk': False,
            'input_text_file': None,
            'output_text_file': None,
            'hmm_model': hmm_path,
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
