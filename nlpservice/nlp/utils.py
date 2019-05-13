import tensorflow as tf
import tensorflow_hub as hub

MODELS = {
    'UniversalSentenceEncoder':
    'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    # 'https://tfhub.dev/google/universal-sentence-encoder/3',
}


def get_model(model, settings=None):
    path = MODELS[model]

    return hub.Module(path)


def sentences2vec(sentences, model=None):
    if model is None:
        model = 'UniversalSentenceEncoder'

    # tf.device('/cpu:0')     # :)
    model = get_model(model)
    init = [tf.global_variables_initializer(), tf.tables_initializer()]

    with tf.Session() as session:
        session.run(init)
        embed = model(sentences)
        vectors = session.run(embed)

    return vectors
