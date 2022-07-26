import time
import requests
import re

wordlist_path = 'dataset/wiki_words/word/ja/jawiki-latest-pages-articles-multistream-index.txt'
podcasturl_path = 'dataset/podcast/feeds/all.txt'
wait_sec = 0.2

chars = sorted(list(set(list(''.join(open(wordlist_path, 'r').readlines())))))
with open(podcasturl_path, 'w') as f:
    for char in chars:
        try:
            url = f'https://podcasts.google.com/search/{char}'
            html = requests.get(url).content

            podcasturls_found = re.findall(r'\./feed/([0-9A-Za-z]+)[^0-9A-Za-z]', str(html))
            podcasturls_found = list(set(podcasturls_found))
            print(f'{char}: {len(podcasturls_found)} {podcasturls_found}')

            f.writelines([v + '\n' for v in podcasturls_found])
            f.flush()
        except:
            print(f'No podcast found for {char}.')

        if wait_sec > 0.01:
            time.sleep(wait_sec)
