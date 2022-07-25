import csv
import os

import requests

with open('dataset/aozora/titles.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    reader.__next__()
    data = [row for row in reader]

with open('dataset/aozora/speakers.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    reader.__next__()
    speaker_dict = {row[1]: int(row[0]) for row in reader}

for row in data:
    title_id = int(row[0])
    if row[3] in speaker_dict:
        speaker_id = speaker_dict[row[3]]
    else:
        speaker_id = 0
    print(title_id, speaker_id, row[3])

    url = f'https://aozoraroudoku.jp/voice/mp3/rd{title_id:03d}.mp3'
    response = requests.get(url)
    if response.status_code != 200:
        break

    os.makedirs(f'dataset/aozora/wav/{speaker_id:03d}', exist_ok=True)
    with open(f'dataset/aozora/wav/{speaker_id:03d}/{title_id:03d}.mp3', 'wb') as f:
        f.write(response.content)
