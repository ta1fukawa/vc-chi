import csv
import re
import os

import lxml.html
import requests

title_id = 1

data = []
while True:
    url = f'https://aozoraroudoku.jp/voice/rdp/rd{title_id:03d}.html'
    response = requests.get(url)
    if response.status_code != 200:
        break

    response.encoding = 'utf-8'
    html = lxml.html.fromstring(response.text)
    title = html.xpath('//*[@id="textBox"]/h1')[0]
    info  = html.xpath('//*[@id="textBox"]/h2')[0]
    author, speaker, time = re.findall(r'^(?:著者：)?(.+)[　 ]読み手：(.+)[　 ]時間：(.+)$', info.text.strip())[0]
    print(title_id, title.text, author, speaker, time)

    data.append([title_id, title.text, author, speaker, time])
    title_id += 1

os.makedirs('dataset/aozora/', exist_ok=True)
with open('dataset/aozora/titles.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'title', 'author', 'speaker', 'time'])
    writer.writerows(data)
