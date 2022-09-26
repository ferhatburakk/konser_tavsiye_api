#!/usr/bin/env python
# coding: utf-8

# ## Loading the Last.fm Data

# The Last.fm data come from the [Music Technology Group](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html) at the [Universitat Pompeu Fabra](http://mtg.upf.edu/) in Barcela, Spain. The data were scraped by Òscar Celma using the Last.fm API, and they are available free of charge. So, thank you Òscar!

# The Last.fm data are broken into two parts: the activity data and the profile data. The activity data is comprised of about 360,000 individual user's Last.fm artist listening information. It details how many times a Last.fm user played songs by various artists. The profile data contains each user's country of residence. We'll use `read.table` from `pandas` to read in the tab-delimited files.

# In[1]:


import json
from fuzzywuzzy import fuzz
import string
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)


user_data = pd.read_table('user_songs.tsv',
                          header=None, nrows=2e7,
                          names=['users', 'musicbrainz-artist-id',
                                 'artist-name', 'plays'],
                          usecols=['users', 'artist-name', 'plays'],
                          )


user_profiles = pd.read_table('user_profiles.tsv',
                              header=None,
                              names=['users', 'gender',
                                     'age', 'country', 'signup'],
                              usecols=['users', 'country'],
                              )


if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis=0, subset=['artist-name'])


artist_plays = (user_data.
                groupby(by=['artist-name'])['plays'].
                sum().
                reset_index().
                rename(columns={'plays': 'total_artist_plays'})
                [['artist-name', 'total_artist_plays']]
                )


user_data_with_artist_plays = user_data.merge(
    artist_plays, left_on='artist-name', right_on='artist-name', how='left')


popularity_threshold = 2000
user_data_popular_artists = user_data_with_artist_plays.query(
    'total_artist_plays >= @popularity_threshold')
user_data_popular_artists.head()


combined = user_data_popular_artists.merge(
    user_profiles, left_on='users', right_on='users', how='left')
tr_data = combined.query('country == \'Turkey\'')
tr_data.head()


if not tr_data[tr_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = tr_data.shape[0]

    print('Initial dataframe shape {0}'.format(tr_data.shape))
    tr_data = tr_data.drop_duplicates(['users', 'artist-name'])
    current_rows = tr_data.shape[0]
    print('New dataframe shape {0}'.format(tr_data.shape))
    print('Removed {0} rows'.format(initial_rows - current_rows))


wide_artist_data = tr_data.pivot(
    index='artist-name', columns='users', values='plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


save_sparse_csr('lastfm_sparse_artist_matrix.npz', wide_artist_data_sparse)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(wide_artist_data_sparse)


wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
wide_artist_data_zero_one_sparse = csr_matrix(wide_artist_data_zero_one.values)

save_sparse_csr('lastfm_sparse_artist_matrix_binary.npz',
                wide_artist_data_zero_one_sparse)


model_nn_binary = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn_binary.fit(wide_artist_data_zero_one_sparse)


def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    """
    Inputs:
    query_artist: query artist name
    artist_plays_matrix: artist play count dataframe (not the sparse one, the pandas dataframe)
    knn_model: our previously fitted sklearn knn model
    k: the number of nearest neighbors.

    Prints: Artist recommendations for the query artist
    Returns: None
    """
    query_index = None
    ratio_tuples = []

    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    print('Possible matches: {0}\n'.format(
        [(x[0], x[1]) for x in ratio_tuples]))

    try:
        # get the index of the best artist match in the data
        query_index = max(ratio_tuples, key=lambda x: x[1])[2]
    except:
        print('Your artist didn\'t match any artists in the data. Try again')
        return None

    distances, indices = knn_model.kneighbors(
        artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=k + 1)

    # for i in range(0, len(distances.flatten())):
    #    if i == 0:

    #       print ('Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index]))
    #   else:
    #       print ('{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i]))
    # return artist_plays_matrix[]
    # print(artist_plays_matrix.index[indices])
    # return artist_plays_matrix.index[indices]

    artistsList = artist_plays_matrix.index[indices][0].tolist()

    return artistsList
