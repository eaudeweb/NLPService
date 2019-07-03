FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="Tiberiu Ichim <tiberiu.ichim@eaudeweb.ro>"

RUN mkdir -p /data
RUN useradd --system -m -d /data -U -u 500 nlp \
	&& chown -R 500 /data

COPY ./ /app

RUN pip3 install -r /app/requirements.txt

RUN pip3 install -e /app

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

COPY docker/* /

VOLUME /data

RUN /docker-setup.sh

EXPOSE 6543
WORKDIR /app

HEALTHCHECK --interval=1m --timeout=5s --start-period=1m \
  CMD nc -z -w5 127.0.0.1 6543 || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["start"]
