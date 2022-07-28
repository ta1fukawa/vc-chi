import requests
import base64
import re

podcasturl_path = 'dataset/podcast/feeds/all.txt'
podcasturl_ja_path = 'dataset/podcast/feeds/ja.txt'
threshold = 0.01

def calc_kana_ratio(string):
    return len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', string)) / len(string)

def base64decode(string):
    return base64.b64decode(string + '=' * (-len(string) % 4)).decode('utf-8')

with open(podcasturl_path, 'r') as f:
    podcasturls = sorted(list(set(f.readlines())))
print('Podcasts:', len(podcasturls))

count = 0
with open(podcasturl_ja_path, 'a') as f:
    for i, podcasturl in enumerate(podcasturls):
        url = base64decode(podcasturl.strip())
        try:
            res = requests.get(url, timeout=(3.0, 3.0))
            if res.status_code != 200:
                raise Exception(f'status code is {res.status_code}')
            rss = res.content.decode(res.apparent_encoding)
            kana_ratio = calc_kana_ratio(str(rss))
            if kana_ratio >= threshold:
                count += 1
                print(f'[{count}/{i+1}] Japanese ({int(kana_ratio * 1000) / 10}%): {url}')
                f.write(podcasturl)
                f.flush()
            else:
                print(f'[{count}/{i+1}] Not Japanese ({int(kana_ratio * 1000) / 10}%): {url}')
        except Exception as e:
            print(f'[{count}/{i+1}] Error: {url} ({e})')
            continue
