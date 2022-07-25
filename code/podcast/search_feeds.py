import time
import requests
import re

wordlist_path = 'dataset/wiki_words/word/ja/jawiki-latest-pages-articles-multistream-index.txt'
podcastid_path = 'dataset/podcastid/all.txt'
wait_sec = 0.2

chars = sorted(list(set(list(''.join(open(wordlist_path, 'r').readlines())))))
with open(podcastid_path, 'w') as f:
    for char in chars:
        try:
            url = f'https://podcasts.google.com/search/{char}'
            html = requests.get(url).content

            podcastids_found = re.findall(r'\./feed/([0-9A-Za-z]+)[^0-9A-Za-z]', str(html))
            podcastids_found = list(set(podcastids_found))
            print(f'{char}: {len(podcastids_found)} {podcastids_found}')

            f.writelines([v + '\n' for v in podcastids_found])
            f.flush()
        except:
            print(f'No podcast found for {char}.')

        if wait_sec > 0.01:
            time.sleep(wait_sec)
