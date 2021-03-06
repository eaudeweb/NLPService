import threading

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

# load one model per thread. This may be inneficient, should research it better
# the problem is that of tensorflow sessions
local = threading.local()


def get_model(name, settings=None):
    if not hasattr(local, 'SESSIONS'):
        local.SESSIONS = {}

    if name not in local.SESSIONS:
        specific_loader, generic_loader = MODELS[name]
        suite = generic_loader(specific_loader)
        local.SESSIONS[name] = suite

    return local.SESSIONS[name]


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
    # TODO: should check if can be used as auto session, automatically choose
    # CPU or GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    keras.backend.set_session(s)

    return s


def gpu_session():
    config = tf.ConfigProto(log_device_placement=True)
    session = tf.Session(config=config)
    keras.backend.set_session(session)

    return session
