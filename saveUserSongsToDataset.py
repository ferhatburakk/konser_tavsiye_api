import csv
from datetime import date
import json

today = date.today()
d2 = today.strftime("%B %d, %Y")
with open('user_songs.json', encoding='utf-8') as fh:
    data = json.load(fh)

for key in data:
    with open('user_profiles.tsv', 'a', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(
            [data[key]['uid'], 'm', '21', 'Turkey', d2])
