""" Main pyramid entry point, console utils
"""

import os

import click
from pyramid.config import Configurator


def main(global_config, **settings):
    config = Configurator(settings=settings)
    config.include("cornice")
    config.scan("nlpservice.views")

    settings = config.get_settings()
    cache_path = settings['nlp.tf_model_cache_path'].strip()
    os.environ['TFHUB_CACHE_DIR'] = cache_path

    return config.make_wsgi_app()
