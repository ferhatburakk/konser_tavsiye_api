import json
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import string
import random
from flask import Flask, Response, jsonify, request
from flask_restful import Resource, Api
import csv
from datetime import date
from last_fm_recommender import print_artist_recommendations
from last_fm_recommender import wide_artist_data_zero_one
from last_fm_recommender import model_nn_binary
from controlConcerts import controlConcertList
from firebaseDatabase import addConcertsToFirebase
app = Flask(__name__)


@app.route('/saveUser', methods=['GET'])
def saveUser():
    if 'userid' in request.args:
        userid = request.args['userid']
        today = date.today()
        d2 = today.strftime("%B %d, %Y")
        with open('user_profiles.tsv', 'a', newline='', encoding='utf-8') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow([userid, 'm', '21', 'Turkey', d2])
            return jsonify(userid), 200


@app.route('/getSongRecommendations', methods=['GET'])
def recommendSongs():
    args = request.args
    userid = args['userid']
    singername = args['singername']
    with open('user_songs.tsv', 'a', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([userid, "newRecord", singername, '2500'])

    data = print_artist_recommendations(
        singername, wide_artist_data_zero_one, model_nn_binary, k=10)

    recommendeds = controlConcertList(data)
    addConcertsToFirebase(recommendeds, userid)
    return jsonify(recommendeds)
    # return jsonify(print_artist_recommendations(
    #    singername, wide_artist_data_zero_one, model_nn_binary, k=10))


if __name__ == "__main__":

    app.run(debug=True)
