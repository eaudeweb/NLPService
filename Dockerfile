FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="Tiberiu Ichim <tiberiu.ichim@eaudeweb.ro>"

RUN mkdir -p /data

COPY ./ /app

RUN pip3 install -r /app/requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

RUN pip3 install -e /app

RUN /usr/local/bin/prepare --es-url http://elasticsearch:9200/content /data/corpus.txt

CMD pserve /app/service.ini
