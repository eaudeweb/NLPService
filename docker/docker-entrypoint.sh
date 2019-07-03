#!/bin/bash
set -e

START="start"
CMD="gunicorn --paster /app/setting.ini -b 0.0.0.0:6543"

# TODO: move these ops to a Makefile
PREPARE="prepare --es-url=http://elasticsearch:9200/content /data/corpus.txt"
LABEL="label /data/corpus.txt /data/labeled-corpus --kg-url=http://nginx/api/knowledge-graph/dump_all/ "
TRAIN_KV="kv /data/corpus.txt /data/corpus-ft"
TRAIN_KERAS="train --cpu data/k-model data/corpus-ft data/corpus.txt --kg-url=http://nginx/api/knowledge-graph/dump_all/"
TRAIN_FT="fasttext supervised -input ./data/labeled-corpus-train -output ./data/labeled-corpus -lr 0.5 -epoch 40 -wordNgrams 2 -bucket 2000000 -dim 50"

if [ ! -e "/data/.models-built" ]; then
  gosu nlp mkdir /data/model_cache
  gosu nlp mkdir /data/cache
  gosu nlp $PREPARE
  gosu nlp $LABEL
  gosu nlp $TRAIN_KV
  gosu nlp $TRAIN_KERAS
  gosu nlp $TRAIN_FT
  gosu nlp touch /data/.models-built
fi

if [[ $START == *"$1"* ]]; then
  exec gosu nlp $CMD
else
  exec "$@"
fi
