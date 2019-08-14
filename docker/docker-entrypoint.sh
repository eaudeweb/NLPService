#!/bin/bash
set -x

START="start"

#CMD="gunicorn --paster /app/production.ini -b 0.0.0.0:6543"
CMD="pserve /app/production.ini"

if [ ! -e "/data/nlp/.models-built" ]; then
  cp /app/docker/Makefile.example /data/nlp/Makefile
  cd /data/nlp/
  chown -R nlp /data/nlp
  gosu nlp make
  gosu nlp touch /data/nlp/.models-built
fi

if [[ $START == *"$1"* ]]; then
  exec gosu nlp $CMD
else
  exec "$@"
fi
