#!/bin/bash
set -e

START="start"
CMD="gunicorn --paster /app/setting.ini -b 0.0.0.0:6543"

GOSU="gosu"

# TODO: move these ops to a Makefile
PREPARE="prepare --es-url=http://elasticsearch:9200/content /data/corpus.txt"
LABEL="label /data/corpus.txt /data/labeled-corpus --kg-url=http://nginx/api/knowledge-graph/dump_all/ "
TRAIN_KV="kv /data/corpus.txt /data/corpus-ft"
TRAIN_KERAS="train --cpu data/k-model data/corpus-ft data/corpus.txt --kg-url=http://nginx/api/knowledge-graph/dump_all/"
TRAIN_FT="fasttext supervised -input ./data/labeled-corpus-train -output ./data/labeled-corpus -lr 0.5 -epoch 40 -wordNgrams 2 -bucket 2000000 -dim 50"

if [ ! -e "/data/nlp/.models-built" ]; then
  $GOSU nlp mkdir /data/model_cache
  $GOSU nlp mkdir /data/cache
  $GOSU nlp $PREPARE
  $GOSU nlp $LABEL
  $GOSU nlp $TRAIN_KV
  $GOSU nlp $TRAIN_KERAS
  $GOSU nlp $TRAIN_FT
  $GOSU nlp touch /data/nlp/.models-built
fi

if [[ $START == *"$1"* ]]; then
  exec gosu nlp $CMD
else
  exec "$@"
fi
