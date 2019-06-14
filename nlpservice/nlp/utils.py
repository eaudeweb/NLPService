import itertools
import logging
from urllib import parse

import click
import requests
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


SESSIONS = {}


def get_model(model, settings=None):
    if model not in SESSIONS:
        path, loader = MODELS[model]
        SESSIONS[model] = loader(path)

    return SESSIONS[model]


def load_use(path):
    import tensorflow as tf
    import tensorflow_hub as hub
    # tf.device('/cpu:0')     # :)
    g = tf.Graph()

    with g.as_default():
        placeholder = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module(path)
        op = embed(placeholder)
        init = [tf.global_variables_initializer(), tf.tables_initializer()]
        init_op = tf.group(init)

    g.finalize()
    session = tf.Session(graph=g)
    session.run(init_op)

    return session, op, placeholder


MODELS = {
    'UniversalSentenceEncoder':
    ('https://tfhub.dev/google/universal-sentence-encoder-large/3', load_use),
}


def sentences2vec(sentences, model=None):
    if model is None:
        model = 'UniversalSentenceEncoder'

    sess, op, placeholder = get_model(model)
    vectors = sess.run(op, feed_dict={placeholder: sentences})

    return vectors


def get_es_records(es_url):
    """ Get all records from an ElasticSearch index
    """

    p = parse.urlparse(es_url)

    server = "{}://{}/".format(p.scheme, p.netloc)

    query = {
        "sort": ["_doc"],
        "size": 1000
    }
    url = "%s/_search?scroll=10m" % (es_url)

    resp = requests.post(url, json=query).json()
    hits = resp["hits"]["hits"]

    # query["scroll"] = "10m"
    query["scroll_id"] = resp["_scroll_id"]

    c = 1000

    def _batch(c):
        logger.info("Scroll position: %s", c)
        q = {
            "scroll": "1m",
            "scroll_id": query["scroll_id"],
            # 'from': cursor
        }
        url = server + "_search/scroll"
        resp = requests.post(url, json=q).json()
        query["scroll_id"] = resp["_scroll_id"]
        c += 1000

        return c, resp["hits"]["hits"]

    while hits:
        yield from (d['_source'] for d in hits)
        c, hits = _batch(c)


def lemmatize_kg_terms(terms):
    """ Lemmatize the terms, to be able to match documents to KG terms
    """
    ps = PorterStemmer()
    out = []

    for t in terms:
        parts = []

        if ',' in t:
            ts = filter(None, [x.strip() for x in t.split(',')])

            for t in ts:
                if t.isupper():
                    continue     # we'll skip acronyms, for now
                doc = word_tokenize(t)
                parts.append([ps.stem(t) for t in doc])
        else:
            doc = word_tokenize(t)
            parts.append([ps.stem(t) for t in doc])

        for part in parts:
            p = tuple([x.lower() for x in part if len(x) > 1])

            if p:
                out.append(p)

    return out
