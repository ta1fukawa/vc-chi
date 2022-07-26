import requests
from xml.etree import ElementTree
import base64
import subprocess
import pathlib
import hashlib

podcasturl_path = pathlib.Path('dataset/podcastid/ja.txt')
# podcasturl_path = pathlib.Path('dataset/podcast/feeds/ja.txt')
dest_path = pathlib.Path('dataset/podcast/ja')

def base64decode(string):
    return base64.b64decode(string + '=' * (-len(string) % 4)).decode('utf-8')

with open(podcasturl_path, 'r') as f:
    podcasturls = f.readlines()

for podcasturl in sorted(podcasturls)[1000:8686]: ###
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
                audio_path = dest_path / 'audio' / podcasturl_hash / '{name}.{ext}'.format(name=audiourl_hash, ext=audiourl.split('.')[-1])
                wav24k_path = dest_path / 'wav24k' / podcasturl_hash / '{name}.wav'.format(name=audiourl_hash)
                if wav24k_path.exists():
                    break
                print(audiourl)

                res = requests.get(audiourl, timeout=(5.0, 5.0))
            except Exception as e:
                print(f'Error: {audiourl} ({e})')
                continue
            if res.status_code != 200:
                print(f'status code is {res.status_code}')
                continue

            audio_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audio_path, 'wb') as f:
                f.write(res.content)
                f.flush()
            print(f'{audiourl} downloaded.')

            wav24k_path = dest_path / 'wav24k' / podcasturl_hash / '{name}.wav'.format(name=audiourl_hash)
            wav24k_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(audio_path), '-ar', '24000', '-ac', '1', str(wav24k_path)])
            print(f'{audiourl} converted to wav24k.')
            print()

            break
        else:
            print(f'{podcasturl} not found.')
            print()
    except Exception as e:
        print(f'Error: {url} ({e})')
        continue
