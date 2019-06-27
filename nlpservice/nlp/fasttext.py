""" Create FastText word embeddings
"""

import logging
from pathlib import Path

import click
from tqdm import tqdm

from gensim.models.fasttext import FastText
from gensim.utils import RULE_DEFAULT, RULE_DISCARD

from .models import get_model

logger = logging.getLogger(__name__)


def trim(word, count, min_count):
    if word == 'dignr':
        return RULE_DISCARD
    else:
        return RULE_DEFAULT


count = 0


def counter(it):
    global count

    for line in it:
        count += 1
        yield line.split(' ')


@click.command()
@click.argument('textfile')
@click.argument('output')
def main(textfile, output):

    logger.setLevel(logging.WARNING)
    model = FastText(size=100, window=3, sg=True, min_count=5,
                     seed=0, word_ngrams=True, trim_rule=trim)

    global count
    count = 0

    with Path(textfile).open() as f:
        lines = (l.strip() for l in f)
        model.build_vocab(sentences=counter(tqdm(lines)))
        f.seek(0)
        model.train(sentences=f, epochs=10, total_examples=count, workers=7)

    model.save(output)


def similar_by_word(word, model):
    ft = get_model(model)
    wv = ft['model'].wv
    similar = wv.similar_by_word(word)
    res = [(x, str(y)) for x, y in similar]

    return res


def load_kv_model(loader):
    """ Generic loader for keyedvectors models.

    :param loader: a callable that returns model path, word embeddings path
                   (for vocabulary) and the labels (classification targets)
    """

    model_path = loader()
    model = FastText.load(model_path)
    vocab = model.wv.index2word

    return {
        'model': model,
        'vocab': vocab,
    }


def corpus_kv_settings(config):
    settings = config.get_settings()

    return settings['nlp.kg_ft_path']
