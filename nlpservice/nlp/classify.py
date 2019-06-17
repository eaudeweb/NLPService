import logging
from collections import Counter, defaultdict, deque

import click
import numpy as np
import requests
from gensim.models import FastText
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Embedding,
                                     GlobalMaxPooling1D, MaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from .prepare import clean
from .utils import lemmatize_kg_terms

logger = logging.getLogger(__name__)


def prepare_corpus(docs, lemmatized_kg):
    """ Get a label for each document. Returns ML compatibile X/y data structs
    """

    unlabeled = 0

    X,  y = [], []

    for doc in docs:
        labels = get_doc_labels(doc, lemmatized_kg)

        if not labels:
            unlabeled += 1

            continue

        X.append(doc)
        y.append(labels)

    return X, y


SPECIAL_TOKENS = {'<pad>': 0, '<start>': 1, '<unk>': 2, '<unused>': 3}


def docs_to_dtm(docs, vocab, maxlen):
    """ Transform docs to term matrixes, padded to maxlen
    """

    # patch the vocabulary to reserve first positions
    token2id = {v: (i + 4) for i, v in enumerate(vocab)}

    def doc2tokens(doc):
        out = []

        for sent in doc:
            words = sent.split(' ')
            out.extend([token2id.get(w, SPECIAL_TOKENS['<unk>'])
                        for w in words])
            out.append(SPECIAL_TOKENS['<start>'])

        return out

    dtm = [doc2tokens(doc) for doc in docs]
    X = pad_sequences(dtm, maxlen=maxlen, padding='post')

    return X


def train_classifier(kvmodel, docs, lemmatized_kg):
    embeddings = kvmodel.wv.vectors
    EMB_DIM = embeddings.shape[1]    # word embedding dimmension
    VOCAB_SIZE = len(embeddings) + len(SPECIAL_TOKENS)
    fill = np.zeros((len(SPECIAL_TOKENS), EMB_DIM))
    emb_vectors = np.vstack((fill, embeddings))

    MAX_LEN = 300      # Max length of text sequences
    X, y = prepare_corpus(docs, lemmatized_kg)

    # one-hot encode labels
    sle = preprocessing.LabelEncoder()
    top_labels = lemmatized_kg.keys()
    sle.fit(list(top_labels))
    y = sle.transform(y)
    y = to_categorical(y, num_classes=len(top_labels))

    X = docs_to_dtm(X, vocab=kvmodel.wv.index2word, maxlen=MAX_LEN)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=0)

    # training params
    batch_size = 100    # 256
    num_epochs = 80

    # model parameters
    num_filters = 64
    weight_decay = 1e-4

    model = Sequential()

    model.add(Embedding(
        VOCAB_SIZE, EMB_DIM, input_length=MAX_LEN,
        weights=[emb_vectors],
        trainable=False
    ))

    # CNN type of network. Accuracy is above 0.95 if used with lower Dropout
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(len(lemmatized_kg), activation='sigmoid'))
    adam = optimizers.Adam(lr=0.001, beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    history = model.fit(
        X_train,
        y_train,
        epochs=num_epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', min_delta=0.01, patience=3, verbose=1)],
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
    )

    loss, accuracy = model.evaluate(X_train, y_train, verbose=True)

    return model


def get_doc_labels(doc, kg):
    found = Counter()

    # TODO: should also stem the doc sentences
    # or parse with Spacy and do word match

    for label, stem_pairs in kg.items():

        for sent in doc:
            for stems in stem_pairs:
                if all([stem in sent for stem in stems]):
                    found[label] += 1

    top = found.most_common()

    if top:
        return top[0][0]

    return []


def terms_from_list(l):
    terms = []

    for ts in l:
        ts = [t.strip() for t in ts.split(',')]
        terms.extend(ts)

    return list(filter(None, terms))


def _flatten(children):

    terms = []
    to_crawl = deque(children)    # kg.values()

    while to_crawl:
        current = to_crawl.popleft()
        terms += [t.strip() for t in current['name'].split(',')]
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


def read_corpus(path):
    """ Returns a list of documents. A doc is a list of sentences.

    A sentence is space separated tokens (words)
    """
    docs = []
    doc = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()

            if not line:
                docs.append(doc)
                doc = []
            else:
                doc.append(line)

    return docs


def stream_corpus(path):
    """ Yields documents. A doc is a list of sentences.

    A sentence is space separated tokens (words)
    """

    doc = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()

            if not line:
                yield doc
                doc = []
            else:
                doc.append(line)


@click.command()
@click.argument('output')
@click.argument('ftpath')
@click.argument('corpus')
@click.option('--kg-url',
              default='http://localhost:8880/api/knowledge-graph/dump_all/',
              help='KnowledgeGraph dump location')
def main(output, ftpath, corpus, kg_url):
    """ Train a Classification model

    :param output: output path for the TF model
    :param ftpath: path for Fasttext word embedding model
    :param corpus: path to a text file of documents. One sentence per line.
                    Separate documents with an empty line
    :
    """
    docs = read_corpus(corpus)

    ft_model = FastText.load(ftpath)

    kg = requests.get(kg_url).json()
    f_kg = flatten_knowledge_graph(kg)
    l_kg = {}

    for b, terms in f_kg.items():
        l_kg[b] = lemmatize_kg_terms(terms)

    k_model = train_classifier(ft_model, docs, l_kg)
    k_model.save(output)

    return output
