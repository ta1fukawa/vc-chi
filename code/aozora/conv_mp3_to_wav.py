import pathlib
# import librosa
import subprocess  # for ffmpeg

mp3_dir = pathlib.Path('dataset/aozora/mp3')
wav_dir = pathlib.Path('dataset/aozorawav')

for mp3_subdir in sorted(mp3_dir.glob('*')):
    wav_subdir = wav_dir / mp3_subdir.name
    wav_subdir.mkdir(parents=True, exist_ok=True)

    for mp3_path in sorted(mp3_subdir.glob('*.mp3')):
        wav_path = wav_subdir / mp3_path.name.replace('.mp3', '.wav')

        if not wav_path.exists():
            print(mp3_path, '->', wav_path)
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(mp3_path), '-ar', '24000', '-ac', '1', str(wav_path)])
