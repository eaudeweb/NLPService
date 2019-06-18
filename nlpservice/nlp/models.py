import threading

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

local = threading.local()


def get_model(model, settings=None):
    if not hasattr(local, 'SESSIONS'):
        local.SESSIONS = {}

    if model not in local.SESSIONS:
        arg, loader = MODELS[model]
        local.SESSIONS[model] = loader(arg)

    return local.SESSIONS[model]


def load_use(path):
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


def nongpu_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    keras.backend.set_session(s)

    return s
