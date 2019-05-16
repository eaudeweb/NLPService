import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.cluster.vq import kmeans2
from sklearn.metrics.pairwise import cosine_similarity

from .utils import sentences2vec


def clusterize_by_topics(text, topics):
    # general approach is to build a map of topicid -> cluster centers for
    # provided seed words

    sentences = []

    for line in text.splitlines():
        # if not line.strip():      # need to keep existing line order
        #     continue              # should probably insert numpy.zeros?
        sentences.extend(sent_tokenize(line.strip()))

    sv = sentences2vec(sentences)

    centers = []
    topic_index = []
    seeds = []

    for i, seedwords in enumerate(topics):
        centers.extend(sentences2vec(seedwords))

        for x, seed in enumerate(seedwords):
            topic_index.append(i)
            seeds.append(seed)

    centers = np.array(centers)
    print(centers.shape)

    c, labels = kmeans2(sv, k=centers, minit='matrix')

    res = []

    for i, (l, sent) in enumerate(zip(labels, sentences)):
        topic = topic_index[l]
        res.append({
            'text': sent,
            'topic': topic,
            'seed': seeds[l]
            # 'confidence': cosine_similarity()
        })

    return res
