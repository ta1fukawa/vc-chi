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
import torch
import torchaudio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

MIN_DURATION_RATIO = 0.3
MAX_SPEAKERS = 3
CHECK_NCHANGES = 3
CHECK_MIN_INTERVAL = 5
MIN_SPEECH_DURATION = 3
MAX_SPEECH_DURATION = 9999
MIN_NSPEECHES = 10
PAD_SIZE = 4000

src_dir = pathlib.Path('dataset/podcast/ja/wav24k')
dst_dir = pathlib.Path('dataset/podcast/ja/wav24k_seg')

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
# seg_model = inaSpeechSegmenter.Segmenter(vad_engine='smn', detect_gender=False, batch_size=1024)

all_nspeakers = 0
x = int(sys.argv[1])
for xx, wav_path in enumerate(sorted(src_dir.glob('*/*'))[x * 300:(x + 1) * 300]):
    print(f'[{xx + x * 300:04d}]', wav_path)
    annotation = pipeline({'audio': wav_path})

    all_duration = 0
    speakers = set(annotation.labels())
    for speaker in speakers:
        all_duration += annotation.label_duration(speaker)
    for speaker in speakers.copy():
        duration = annotation.label_duration(speaker)
        if duration / all_duration < MIN_DURATION_RATIO:
            speakers.remove(speaker)
    print(f'  Speakers: {speakers}')

    if len(speakers) >= 1:
        segments = [(segment[2], segment[0], True) for segment in list(annotation.itertracks(yield_label=True))]  # (speaker, turn, flag)
        speaker_changes = [('_', -CHECK_MIN_INTERVAL, -1) for _ in range(CHECK_NCHANGES)]  # (speaker, end_time, segment_index)
        for i in range(len(segments)):
            speaker, turn, flag = segments[i]
            # print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')

            if speaker != speaker_changes[-1][0]:
                speaker_changes.append((speaker, turn.start, i))
            if speaker_changes[-CHECK_NCHANGES][1] > turn.end - CHECK_MIN_INTERVAL:
                for j in range(speaker_changes[-CHECK_NCHANGES][2], i + 1):
                    segments[j] = (segments[j][0], segments[j][1], False)
                # print(f'  {speaker_changes[-check_nchanges][1]} <= {turn.end} - {CHECK_MIN_INTERVAL}')
                continue
            if speaker not in speakers:
                segments[i] = (speaker, turn, False)
                # print(f'  {speaker} is not in {speakers}')
                continue

        ok_segments = []
        for speaker, turn, flag in segments:
            if flag and MAX_SPEECH_DURATION >= turn.duration >= MIN_SPEECH_DURATION:
                ok_segments.append((speaker, turn))

        speaker_segments = {}
        for target_speaker in speakers:
            speaker_segments[target_speaker] = []
        for speaker, turn in ok_segments:
            speaker_segments[speaker].append(turn)
        for target_speaker in list(speaker_segments.keys()):
            if len(speaker_segments[target_speaker]) < MIN_NSPEECHES:
                del speaker_segments[target_speaker]
        if len(speaker_segments) > MAX_SPEAKERS or len(speaker_segments) == 0:
            print(f'  Too many or too few speakers: {len(speaker_segments)}')
            continue
        all_nspeakers += len(speaker_segments)

        wave, sr = torchaudio.load(wav_path)
        wave /= wave.max()
        wave = wave.squeeze().numpy()

        for target_speaker in speaker_segments.keys():
            dst_speaker_dir = dst_dir / f'{wav_path.parent.stem}_{wav_path.stem}_{target_speaker}'
            dst_speaker_dir.mkdir(parents=True, exist_ok=True)

            for i, turn in enumerate(speaker_segments[target_speaker]):
                speech_path = dst_speaker_dir / f'podcast_{i:03d}.wav'
                sf.write(speech_path, np.pad(wave[int(turn.start * sr):int(turn.end * sr)], (PAD_SIZE, PAD_SIZE), 'constant'), sr)

    # elif len(speakers) == 1:
        # segments = seg_model(wav_path)

        # wave, sr = torchaudio.load(wav_path)
        # wave /= wave.max()
        # wave = wave.squeeze().numpy()

        # dst_speaker_dir = dst_dir / f'{wav_path.parent.stem}_{wav_path.stem}_A'
        # dst_speaker_dir.mkdir(parents=True, exist_ok=True)

        # segments = [(state, start, end) for state, start, end in segments if state == 'speech' and MAX_SPEECH_DURATION >= end - start >= MIN_SPEECH_DURATION]
        # if len(segments) < MIN_NSPEECHES:
        #     print(f'  Too few speech segments: {len(segments)}')
        #     continue

        # for i, (state, start, end) in enumerate(segments):
        #     speech_path = dst_speaker_dir / f'podcast_{i:03d}.wav'
        #     sf.write(speech_path, np.pad(wave[int(start * sr):int(end * sr)], (PAD_SIZE, PAD_SIZE), 'constant'), sr)
    else:
        print(f'  No speakers')
        continue
