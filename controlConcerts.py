# Önerilen şarkıcıların konserlerinin olup olmadığının kontrolü
import json
from urllib import response
from flask import jsonify
import requests

header_dict = {"X-Etkinlik-Token": "a8a3969a6531cecd6ff919a4e38a3b24"}

response = requests.get(
    "https://backend.etkinlik.io/api/v2/events?format_ids=19", headers=header_dict)

json_response = response.json()

# konserleri yazdırma
# for data in json_response['items']:
#    print(data['name'])

# önerilen şarkıcıların konseri var mı?
def controlConcertList(artists):
    recommendedConcerts = []
    for key1 in artists:
        for key2 in json_response['items']:
            if key1 == key2['name']:
                #print(key2)
                recommendedConcerts.append(key2) #name kısmı silinecek
    return recommendedConcerts
