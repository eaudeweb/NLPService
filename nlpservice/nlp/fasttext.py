""" Create FastText word embeddings
"""

import logging
from pathlib import Path

import click
from gensim.models.fasttext import FastText
from gensim.utils import RULE_DEFAULT, RULE_DISCARD
from tqdm import tqdm

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
        model.build_vocab(sentences=counter(tqdm(f)))
        f.seek(0)
        model.train(sentences=f, epochs=10, total_examples=count, workers=7)

    model.save(output)
