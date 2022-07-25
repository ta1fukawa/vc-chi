import pathlib
import shutil
import zipfile

import soundfile as sf


# RUN: gdown 19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt

zip_path = pathlib.Path('./dataset/jvs_ver1.zip')

## Extract zip file
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    zip_file.extractall(zip_path.parent)

jvs_dir = pathlib.Path('./dataset', zip_path.stem)
wav_dir = jvs_dir / 'wav'
all_dir = jvs_dir / 'wav_all'

speakers = sorted(jvs_dir.iterdir())

wav_dir.mkdir(exist_ok=True)


## Relocate wav files
for speaker in speakers:
    if not speaker.is_dir():
        speaker.unlink()
        continue

    shutil.move(str(speaker / 'parallel100' / 'wav24kHz16bit'), str(wav_dir / speaker.name))
    shutil.rmtree(str(speaker))

## Fix error wav files
# Move
for speech in range(21, 14, -1):
    shutil.move(
        str(wav_dir / f'jvs058' / f'VOICEACTRESS100_{speech:03d}.wav'), 
        str(wav_dir / f'jvs058' / f'VOICEACTRESS100_{speech + 1:03d}.wav')
    )
# Split
wave, sr = sf.read(str(wav_dir / f'jvs058' / f'VOICEACTRESS100_014.wav'))
sf.write(str(wav_dir / f'jvs058' / f'VOICEACTRESS100_014.wav'), wave[:int(sr * 3.53)], sr, subtype='PCM_16')
sf.write(str(wav_dir / f'jvs058' / f'VOICEACTRESS100_015.wav'), wave[int(sr * 3.94):], sr, subtype='PCM_16')


shutil.copytree(str(wav_dir), str(all_dir))


## Wav: Fix error wav files
# Remove
speakers = [9, 17, 18, 22, 24, 36, 38, 43, 47, 48, 51, 55, 58, 59, 60, 74, 98]
for speaker in speakers:
    shutil.rmtree(str(wav_dir / f'jvs{speaker:03d}'))
# Remove
speeches = [6, 19, 25, 41, 43, 45, 47, 56, 57, 60, 61, 62, 64, 66, 72, 74, 76, 82, 85, 86, 88, 94, 95, 99]
for speaker in sorted(wav_dir.iterdir()):
    for speech in speeches:
        if (speaker / f'VOICEACTRESS100_{speech:03d}.wav').exists():
            (speaker / f'VOICEACTRESS100_{speech:03d}.wav').unlink()


## Bak: Fix error wav files
# Remove
ss = [(9, 86), (9, 95), (17, 82), (18, 72), (22, 47), (24, 88), (36, 57), (38, 6), (38, 41), (43, 85), (47, 85), (48, 43), (48, 76), (51, 25), (55, 56), (55, 76), (55, 99), (58, 14), (59, 61), (59, 64), (59, 66), (59, 74), (60, 82), (74, 62), (98, 60), (98, 99)]
for speaker, speech in ss:
    if (all_dir / f'jvs{speaker:03d}' / f'VOICEACTRESS100_{speech:03d}.wav').exists():
        (all_dir / f'jvs{speaker:03d}' / f'VOICEACTRESS100_{speech:03d}.wav').unlink()
