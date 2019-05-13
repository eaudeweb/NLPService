import math

import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .utils import sentences2vec


def summarize_text(text, target_len=None, keep=None):
    keep = keep or []
    sentences = sent_tokenize(text)

    if not target_len:
        target_len = math.ceil((math.sqrt(len(sentences)))) or 1

    assert target_len <= len(sentences)

    sent_v = sentences2vec(sentences)
    k_model = KMeans(n_clusters=target_len, random_state=0)
    k_model = k_model.fit(sent_v)

    summary = []

    # get a list of indexes: for each center, which is the index for the
    # closest sentence to that center
    closest, _ = pairwise_distances_argmin_min(k_model.cluster_centers_,
                                               sent_v)

    # given a list of clusters, find which sentence sits in the middle,
    # for the sentences belonging to that cluster
    avg = []

    for j in range(target_len):
        idx = np.where(k_model.labels_ == j)[0]
        avg.append(np.mean(idx))
    # lookup "table" for averages
    ordering = sorted(range(target_len), key=lambda k: avg[k])
    sum_sents = [sentences[closest[i]] for i in ordering]

    summary = {
        'preview': '\n'.join(sum_sents),
        'sentences': [],
    }

    for i, sent in enumerate(sentences):
        line = {
            'text': sent,
            'is_summary': i in closest,
            'keep': i in keep,
        }
        summary['sentences'].append(line)

    return summary
