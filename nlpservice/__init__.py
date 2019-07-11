""" Main pyramid entry point, console utils
"""

import logging
import os

from nlpservice.nlp.classify import load_classifier_model
from nlpservice.nlp.fasttext import load_kv_model
from nlpservice.nlp.models import MODELS
from pyramid.config import Configurator
from pyramid.util import DottedNameResolver

from .utils import get_keys_by_prefix

logger = logging.getLogger(__name__)


def prepare_model_loaders(config, prefix, model_loader):
    """
    """

    def config_wrapper(factory):
        def inner():
            try:
                return factory(config)
            except:
                logger.exception("Could not run model factory %r", factory)

        return inner

    settings = config.get_settings()

    for k, v in get_keys_by_prefix(settings, prefix):

        name = k.rsplit('.', 1)[1]

        for dottedname in v.split():
            factory = DottedNameResolver().resolve(dottedname)
            logger.warning('Resolved %s: %s', dottedname, factory)

            MODELS[name] = (config_wrapper(factory), model_loader)

    print(MODELS)


def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.include("cornice")
    config.scan("nlpservice.views")

    settings = config.get_settings()
    cache_path = settings['nlp.tf_model_cache_path'].strip()
    os.environ['TFHUB_CACHE_DIR'] = cache_path

    prepare_model_loaders(config, 'nlp.classifier.', load_classifier_model)
    prepare_model_loaders(config, 'nlp.keyedvectors.', load_kv_model)

    return config.make_wsgi_app()
