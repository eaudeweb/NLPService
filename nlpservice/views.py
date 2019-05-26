""" Cornice services.
"""

from colander import Int, Length, MappingSchema, SchemaNode, String
from cornice import Service
from cornice.validators import colander_body_validator

from .nlp.cluster import clusterize_by_topics
from .nlp.duplicate import duplicate_detection
from .nlp.prepare import clean
from .nlp.similarity import semantic_similarity
from .nlp.summarize import summarize_text

hello = Service(name='hello', path='/', description="Simplest app")


@hello.get()
def get_info(request):
    """Returns Hello in JSON."""

    return {'Hello': 'World'}


summarize = Service(
    name='summarize', path='/summarize',
    description='Simple text summarization service',
    cors_enabled=True, cors_origins="*",
)


class SummarizeSchema(MappingSchema):
    text = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=100, max=10000)
    )
    target_length = SchemaNode(Int(), missing=0)


@summarize.post(schema=SummarizeSchema, validators=(colander_body_validator))
def summarize_text_view(request):
    text = request.validated['text']
    summary = summarize_text(text)

    return {
        'result': summary
    }


similarity = Service(
    name='similarity', path='/similarity',
    description='Simple text similarity service',
    cors_enabled=True, cors_origins="*",
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


@similarity.post(schema=SimilaritySchema, validators=(colander_body_validator))
def similarity_text_view(request):
    base = request.validated['base']
    proba = request.validated['proba']

    score = semantic_similarity(base, proba)

    return {
        'result': str(score)
    }


# This doesn't handle duplicates, only sentences
duplicate = Service(
    name='duplicates', path='/duplicates',
    description='Simple sentence duplicate detection service',
    cors_enabled=True, cors_origins="*",
)


class DuplicateSchema(MappingSchema):
    text = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=100, max=10000)
    )


@duplicate.post(schema=DuplicateSchema, validators=(colander_body_validator))
def duplicate_text_view(request):
    text = request.validated['text']

    duplicates = duplicate_detection(text)

    return {
        'result': duplicates
    }


clusterize = Service(
    name='clusterize', path='/clusterize',
    description='Clusterize sentences according to seed topics',
    cors_enabled=True, cors_origins="*",
)


class ClusterizeSchema(MappingSchema):
    text = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=100, max=10000)
    )
    topics = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=10, max=10000)
    )


@clusterize.post(schema=ClusterizeSchema, validators=(colander_body_validator))
def clusterize_text_view(request):
    text = request.validated['text']
    topics = request.validated['topics']

    st = []

    for t in topics.split('\n'):
        ts = [x.strip() for x in t.split(',')]
        st.append(ts)

    clusters = clusterize_by_topics(text, st)

    return {
        'result': clusters
    }


prepare = Service(
    name='prepare', path='/prepare',
    description='Prepare text to be further processed. Mostly cleanup.',
    cors_enabled=True, cors_origins="*",
)


class PrepareSchema(MappingSchema):
    text = SchemaNode(
        String(encoding='utf-8', allow_empty=False),
        validator=Length(min=10, max=100000)
    )


@prepare.post(schema=PrepareSchema, validators=(colander_body_validator))
def prepare_text_view(request):
    text = request.validated['text']

    result = clean(text)

    return {
        'result': result,
    }
