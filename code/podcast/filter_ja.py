import requests
import base64
import re

podcastid_path = 'dataset/podcastid/all.txt'
podcastid_ja_path = 'dataset/podcastid/ja.txt'

def is_hiragana_used(string):
    return re.search(r'[\u3040-\u309F]', string) is not None

def base64decode(string):
    return base64.b64decode(string + '=' * (-len(string) % 4)).decode('utf-8')

with open(podcastid_path, 'r') as f:
    podcastids = sorted(list(set(f.readlines())))
print('Podcasts:', len(podcastids))

count = 0
with open(podcastid_ja_path, 'w') as f:
    for i, podcastid in enumerate(podcastids):
        url = base64decode(podcastid.strip())
        try:
            res = requests.get(url)
            if res.status_code != 200:
                raise Exception(f'status code is {res.status_code}')
            rss = res.content.decode(res.apparent_encoding)
            if is_hiragana_used(str(rss)):
                count += 1
                print(f'[{count}/{i+1}] May possibly be Japanese: {url}')
                f.write(podcastid)
                f.flush()
            else:
                print(f'[{count}/{i+1}] Will not be Japanese: {url}')
        except Exception as e:
            print(f'[{count}/{i+1}] Error: {url} ({e})')
            continue
