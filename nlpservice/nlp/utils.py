import itertools
import logging
from collections import defaultdict, deque
from urllib import parse

import click
import requests

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from .models import get_model

logger = logging.getLogger(__name__)


def sentences2vec(sentences, model=None):
    if model is None:
        model = 'UniversalSentenceEncoder'

    sess, op, placeholder = get_model(model)
    vectors = sess.run(op, feed_dict={placeholder: sentences})

    return vectors


def get_es_records(es_url, batch_size=1000):
    """ Get all records from an ElasticSearch index
    """

    p = parse.urlparse(es_url)

    server = "{}://{}/".format(p.scheme, p.netloc)

    query = {
        "sort": ["_doc"],
        "size": batch_size
    }
    url = "%s/_search?scroll=10m" % (es_url)

    resp = requests.post(url, json=query).json()
    hits = resp["hits"]["hits"]

    query["scroll_id"] = resp["_scroll_id"]

    c = batch_size

    def _batch(c):
        logger.info("Scroll position: %s", c)
        q = {
            "scroll": "1m",
            "scroll_id": query["scroll_id"],
        }
        url = server + "_search/scroll"
        resp = requests.post(url, json=q).json()
        query["scroll_id"] = resp["_scroll_id"]
        c += batch_size

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


def _flatten(children):

    terms = []
    to_crawl = deque(children)    # kg.values()

    while to_crawl:
        current = to_crawl.popleft()
        # terms += [t.strip() for t in current['name'].split(',')]
        # don't use the abbreviations
        terms += [current['name'].split(',')[0]]
        to_crawl.extend(current.get('children', []))

    return terms


def flatten_knowledge_graph(kg):
    """ Flatten the KG to a map of major branches and their trigger words
    """

    res = defaultdict(list)

    for mj_branch in kg:
        name = mj_branch['name']
        res[name] = [name] + _flatten(mj_branch.get('children', []))

    return res


def get_lemmatized_kg(url):
    kg = requests.get(url).json()
    flat = flatten_knowledge_graph(kg)
    res = {}

    for b, terms in flat.items():
        res[b] = lemmatize_kg_terms(terms)

    return res


def terms_from_list(l):
    # unused?
    terms = []

    for ts in l:
        ts = [t.strip() for t in ts.split(',')]
        terms.extend(ts)

    return list(filter(None, terms))
