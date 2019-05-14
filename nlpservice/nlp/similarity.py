from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .utils import sentences2vec


def semantic_similarity(base, proba):
    sents_b = sent_tokenize(base)
    sents_p = sent_tokenize(proba)

    v_b = sentences2vec(sents_b)
    v_p = sentences2vec(sents_p)

    k_b = KMeans(n_clusters=1).fit(v_b)
    k_p = KMeans(n_clusters=1).fit(v_p)

    c_b = k_b.cluster_centers_[0]
    c_p = k_p.cluster_centers_[0]

    score = cosine_similarity([c_b], [c_p])[0, 0]

    return score
