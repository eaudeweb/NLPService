import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    README = f.read()


install_requires = [
    'cornice',
    'waitress',
    'tensorflow-hub',
    'colander',
]


setup(name='nlpservice',
      version=0.1,
      description='NLP Services via REST',
      long_description=README,
      classifiers=[
          "Programming Language :: Python",
          "Framework :: Pylons",
          "Topic :: Internet :: WWW/HTTP",
          "Topic :: Internet :: WWW/HTTP :: WSGI :: Application"
      ],
      keywords="web services",
      author='',
      author_email='',
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points="""\
      [paste.app_factory]
      main=nlpservice:main
      """,
      paster_plugins=['pyramid'])
