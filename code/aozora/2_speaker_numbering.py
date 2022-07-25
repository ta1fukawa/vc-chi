import csv

with open('dataset/aozora/titles.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    reader.__next__()
    data = [row for row in reader]

speakers = list(set([row[3] for row in data if '、' not in row[3]]))
speaker_ids = [(i + 1, speaker) for i, speaker in enumerate(speakers)]

with open('dataset/aozora/speakers.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'name'])
    writer.writerows(speaker_ids)
