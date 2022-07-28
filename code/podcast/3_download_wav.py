import base64
import hashlib
import os
import pathlib
import random
import shutil
import subprocess
import sys
import warnings
from contextlib import redirect_stdout
from xml.etree import ElementTree

import requests

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', UserWarning)

import inaSpeechSegmenter
import numpy as np
import tensorflow as tf
import torchaudio

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

podcasturl_path = pathlib.Path('dataset/podcast/feeds/ja.txt')
dest_path = pathlib.Path('dataset/podcast/ja')

def base64decode(string):
    return base64.b64decode(string + '=' * (-len(string) % 4)).decode('utf-8')

def download(url, path):
    res = requests.get(audiourl, timeout=(5.0, 5.0))
    if res.status_code != 200:
        return False
    with open(path, 'wb') as f:
        f.write(res.content)
        f.flush()
    return True

def convert_to_wav24k(src, dst):
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(src), '-ar', '24000', '-ac', '1', str(dst)])

min_speech_ratio = 0.80
max_music_ratio  = 0.0100
max_noise_mean_level = 0.0100

seg_model = inaSpeechSegmenter.Segmenter(vad_engine='smn', detect_gender=False, batch_size=1024)

def check_quality(path):
    with redirect_stdout(open(os.devnull, 'w')):
        data = seg_model(path)

    wave, sr = torchaudio.load(path)
    wave /= wave.max()
    wave = wave.squeeze().numpy()

    speech_duration = 0
    music_duration  = 0
    noise_duration  = 0
    noise_wave = []
    for state, start, end in data:
        duration = end - start
        if state == 'speech':
            speech_duration += duration
        elif state == 'music':
            music_duration += duration
        elif state == 'noise':
            noise_duration += duration
            noise_wave.append(wave[int(start * sr):int(end * sr)])

    full_duration = speech_duration + music_duration + noise_duration
    speech_ratio = speech_duration / full_duration
    music_ratio  = music_duration / full_duration
    noise_mean_level = np.mean(np.abs(np.concatenate(noise_wave))) if noise_wave else -np.inf

    print('Analysis: speech_ratio={:.2f}, music_ratio={:.4f}, noise_mean_level={:.4f})'.format(speech_ratio, music_ratio, noise_mean_level))

    if speech_ratio > min_speech_ratio and music_ratio < max_music_ratio and noise_mean_level < max_noise_mean_level:
        return True
    else:
        return False

print()

with open(podcasturl_path, 'r') as f:
    podcasturls = f.readlines()

rand_fname = ''.join(random.choice('0123456789abcdef') for _ in range(8))
for i, podcasturl in enumerate(sorted(podcasturls)[5669:5700]):
    print(f'[{i+5669}/{len(podcasturls)}] {podcasturl}')
    podcasturl_hash = hashlib.md5(podcasturl.strip().encode()).hexdigest()

    try:
        url = base64decode(podcasturl.strip())
        res = requests.get(url, timeout=(3.0, 3.0))
        if res.status_code != 200:
            raise Exception(f'status code is {res.status_code}')
        rss = res.content.decode(res.apparent_encoding)

        data = ElementTree.fromstring(rss)
        for item in data.findall('channel/item'):
            try:
                audiourl = item.find('enclosure').get('url')
                audiourl_hash = hashlib.md5(audiourl.encode()).hexdigest()
                print(audiourl)

                audio_path  = pathlib.Path('/tmp', '{fname}.{ext}'.format(fname=rand_fname, ext=audiourl.split('.')[-1]))
                if download(audiourl, audio_path):
                    print(f'Downloaded.')
                else:
                    print(f'Download failed.')
                    continue

                audio24k_path = pathlib.Path('/tmp', '{fname}_24k.wav'.format(fname=rand_fname))
                convert_to_wav24k(audio_path, audio24k_path)
                print(f'Converted to wav24k.')

                if check_quality(audio24k_path):
                    wav_path = dest_path / 'wav24k' / podcasturl_hash / '{name}.wav'.format(name=audiourl_hash)
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(audio24k_path, wav_path)
                    print(f'Copied to {wav_path}.')
                else:
                    print(f'Not good quality.')

                audio_path.unlink(missing_ok=True)
                audio24k_path.unlink(missing_ok=True)
            except Exception as e:
                continue
            break
        else:
            print(f'Not found.')
    except Exception as e:
        continue

#  find dataset/podcast/ja/wav24k -type d -empty -delete
