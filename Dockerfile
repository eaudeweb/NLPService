FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="Tiberiu Ichim <tiberiu.ichim@eaudeweb.ro>"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gosu \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data
RUN useradd --system -m -d /data -U -u 500 nlp \
	&& chown -R 500 /data

COPY . /app

RUN pip3 install --no-cache-dir -r /app/requirements.txt

RUN pip3 install -e /app

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt

VOLUME /data/nlp

# RUN /app/docker/docker-setup.sh

EXPOSE 6543
WORKDIR /app

HEALTHCHECK --interval=1m --timeout=5s --start-period=1m \
  CMD nc -z -w5 127.0.0.1 6543 || exit 1

#ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]

#CMD ["start"]
CMD ["bash"]
