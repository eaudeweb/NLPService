""" Main pyramid entry point, console utils
"""

import os

from pyramid.config import Configurator
from pyramid.util import DottedNameResolver

from nlpservice.nlp.models import MODELS


def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.include("cornice")
    config.scan("nlpservice.views")

    settings = config.get_settings()
    cache_path = settings['nlp.tf_model_cache_path'].strip()
    os.environ['TFHUB_CACHE_DIR'] = cache_path

    for k, v in settings.items():
        if not k.startswith('nlp.classifiers'):
            continue
        name = k.rsplit('.')[1]

        for dottedname in v.split():
            func = DottedNameResolver().resolve(dottedname)
            MODELS[name] = func(config)

    return config.make_wsgi_app()
