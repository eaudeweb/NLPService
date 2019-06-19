FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="Tiberiu Ichim <tiberiu.ichim@eaudeweb.ro>"

COPY ./ /app
RUN pip3 install -e /app

RUN python -m nltk.downloader stopwords

CMD pserve /app/service.ini
