""" Cornice services.
"""

from colander import Int, Length, MappingSchema, SchemaNode, String
from cornice import Service
from cornice.validators import colander_body_validator

from .nlp.summarize import summarize_text

hello = Service(name='hello', path='/', description="Simplest app")


@hello.get()
def get_info(request):
    """Returns Hello in JSON."""

    return {'Hello': 'World'}


summarize = Service(name='summarize', path='/summarize',
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
        'summary': summary
    }
