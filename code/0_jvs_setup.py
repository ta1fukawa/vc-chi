import pathlib
import shutil

import soundfile as sf


jvs_dir = pathlib.Path('./dataset/jvs_ver1')
wav_dir = jvs_dir / 'wav'

speakers = sorted(jvs_dir.iterdir())

wav_dir.mkdir(exist_ok=True)

## Relocate wav files

for speaker in speakers:
    if not speaker.is_dir():
        speaker.unlink()
        continue

    shutil.move(str(speaker / 'parallel100' / 'wav24kHz16bit'), str(wav_dir / speaker.name))
    shutil.rmtree(str(speaker))

## Fix or remove error wav files

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
