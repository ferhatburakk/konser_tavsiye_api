import json
import firebase_admin
from firebase_admin import credentials, firestore

dbinfo = credentials.Certificate("firebase_info.json")
firebase_admin.initialize_app(dbinfo)

firestoreDb = firestore.client()


def addConcertsToFirebase(concerts, uid):
    datas = concerts
    for key in datas:
        artist = key['name']
        info = key['content']
        poster_url = key['poster_url']
        date = key['start']
        ticket_url = key['ticket_url']
        id = key['id']
        firestoreDb.collection(u'user_concerts').add(
            {'uid': uid, 'id': id, 'artist': artist, 'info': info, 'poster_url': poster_url, 'date': date, 'ticket_url': ticket_url})
