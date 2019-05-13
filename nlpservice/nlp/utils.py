import tensorflow as tf
import tensorflow_hub as hub

SESSIONS = {}


def get_model(model, settings=None):
    if model not in SESSIONS:
        path, loader = MODELS[model]
        SESSIONS[model] = loader(path)

    return SESSIONS[model]


def load_use(path):
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
    # 'https://tfhub.dev/google/universal-sentence-encoder/3',
}


def sentences2vec(sentences, model=None):
    if model is None:
        model = 'UniversalSentenceEncoder'

    sess, op, placeholder = get_model(model)
    vectors = sess.run(op, feed_dict={placeholder: sentences})

    return vectors
