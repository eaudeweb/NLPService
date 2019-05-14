from collections import defaultdict

import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from .utils import sentences2vec


def intercluster_similarity(vectors):
    simil = 1 - cosine_distances(vectors)

    return np.mean(simil)


def duplicate_detection(text):
    sentences = []

    for line in text.splitlines():
        # if not line.strip():      # need to keep existing line order
        #     continue              # should probably insert numpy.zeros?
        sentences.extend(sent_tokenize(line.strip()))

    sv = sentences2vec(sentences)
    # ideally eps should be tested for the data
    # this value detects close duplicates
    dm = DBSCAN(eps=0.1, min_samples=2, metric='cosine')
    labels = dm.fit_predict(sv)
    res = []

    by_cluster = defaultdict(list)

    for i, l in enumerate(labels):
        c_v = sv[i]
        by_cluster[l].append(c_v)

    similarities = {}

    for k, vects in by_cluster.items():
        a = np.array(vects)
        score = intercluster_similarity(a)
        similarities[k] = score

    for c, t in zip(labels, sentences):
        info = {
            'text': t,
            'c': str(c),
            'score': str(similarities[c])
        }
        res.append(info)

    return res
