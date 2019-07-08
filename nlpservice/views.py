""" Cornice services.
"""

from colander import (Int, Length, MappingSchema, SchemaNode, SequenceSchema,
                      String)
from cornice import Service
from cornice.resource import resource, view
from cornice.validators import colander_body_validator

from .nlp.classify import predict_classes
from .nlp.cluster import clusterize_by_topics
from .nlp.duplicate import duplicate_detection
from .nlp.fasttext import similar_by_word
from .nlp.prepare import clean
from .nlp.similarity import semantic_similarity
from .nlp.summarize import summarize_text

summarize = Service(
    name='summarize', path='/summarize',
    description='Simple text summarization service',
    cors_enabled=True, cors_origins="*",
)


class SimpleTextSchema(MappingSchema):
    text = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=2, max=100000)
    )


class Models(SequenceSchema):
    model = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=2, max=100)
    )


class ModelListSchema(MappingSchema):
    models = Models()


class SummarizeSchema(SimpleTextSchema):
    target_length = SchemaNode(Int(), missing=0)


class SimpleModelSchema(SimpleTextSchema):
    model = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        # TODO: write validator
    )


class SimilaritySchema(MappingSchema):
    base = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=100, max=10000)
    )
    proba = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=100, max=10000)
    )


class ClusterizeSchema(SimpleTextSchema):
    topics = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=10, max=10000)
    )


@summarize.post(schema=SummarizeSchema, validators=(colander_body_validator))
def summarize_text_view(request):
    text = request.validated['text']

    return {
        'result': summarize_text(text)
    }


similarity = Service(
    name='similarity', path='/similarity',
    description='Simple text similarity service',
    cors_enabled=True, cors_origins="*",
)


@similarity.post(schema=SimilaritySchema, validators=(colander_body_validator))
def similarity_text_view(request):
    base = request.validated['base']
    proba = request.validated['proba']

    return {
        'result': str(semantic_similarity(base, proba))
    }


# This doesn't handle duplicates, only sentences
duplicate = Service(
    name='duplicates', path='/duplicates',
    description='Simple sentence duplicate detection service',
    cors_enabled=True, cors_origins="*",
)


@duplicate.post(schema=SimpleTextSchema, validators=(colander_body_validator))
def duplicate_text_view(request):
    text = request.validated['text']

    return {
        'result': duplicate_detection(text)
    }


clusterize = Service(
    name='clusterize', path='/clusterize',
    description='Clusterize sentences according to seed topics',
    cors_enabled=True, cors_origins="*",
)


@clusterize.post(schema=ClusterizeSchema, validators=(colander_body_validator))
def clusterize_text_view(request):
    text = request.validated['text']
    topics = request.validated['topics']

    st = []

    for t in topics.split('\n'):
        ts = [x.strip() for x in t.split(',')]
        st.append(ts)

    return {
        'result': clusterize_by_topics(text, st)
    }


prepare = Service(
    name='prepare', path='/prepare',
    description='Prepare text to be further processed. Mostly cleanup.',
    cors_enabled=True, cors_origins="*",
)


@prepare.post(schema=SimpleTextSchema, validators=(colander_body_validator))
def prepare_text_view(request):
    text = request.validated['text']

    return {
        'result': clean(text)
    }


kv_synonyms = Service(
    name='kv_synonyms', path='/kv_synonyms',
    description='Yield synonyms based on KeyedVectors model',
    cors_enabled=True, cors_origins="*",
)


@kv_synonyms.post(schema=SimpleModelSchema,
                  validators=(colander_body_validator))
def kv_synonyms_view(request):
    text = request.validated['text']
    model = request.validated['model']

    return {
        'result': similar_by_word(text, model)
    }


@resource(collection_path="/classify/", path="/classify/{id}",
          description='Run classification predictions on text',
          cors_enabled=True, cors_origins="*",
          )
class Classifier(object):
    def __init__(self, request, context=None):
        self.context = context
        self.request = request

    def _get_model_names(self):
        from nlpservice import get_keys_by_prefix
        pairs = get_keys_by_prefix(self.request.registry.settings,
                                   'nlp.classifier.')
        names = [p[0] for p in pairs]
        names = [n.rsplit('.', 1)[1] for n in names]

        return list(set(names))

    def collection_get(self):
        return {
            'result': self._get_model_names()
        }

    @view(schema=SimpleTextSchema, validators=(colander_body_validator))
    def post(self):
        request = self.request
        text = request.validated['text']
        model = request.matchdict['id']

        return {
            'result': predict_classes(text, model)
        }

    @view(schema=ModelListSchema, validators=(colander_body_validator))
    def collection_post(self):
        """ Retrain configured models
        """
        from .nlp.models import get_model

        names = self._get_model_names()
        train = [t for t in names if t in self.request.validated['models']]

        for name in train:
            model = get_model(name)
            model['train']()

        return {
            'result': 'ok'
        }
