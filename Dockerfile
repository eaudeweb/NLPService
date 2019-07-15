FROM tensorflow/tensorflow:latest-py3
LABEL maintainer="Tiberiu Ichim <tiberiu.ichim@eaudeweb.ro>"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        gosu \
        vim \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN git clone https://github.com/facebookresearch/fastText.git /tmp/fastText && \
  rm -rf /tmp/fastText/.git* && \
  mv /tmp/fastText /app && \
  cd /app/fastText && \
  make

VOLUME /data/nlp

RUN mkdir -p /data/nltk_data
RUN useradd --system -m -d /data -U -u 500 nlp \
	&& chown -R 500 /data

RUN pip3 install --no-cache-dir -r /app/requirements.txt

RUN chown -R 500 /app
RUN pip3 install -e /app

RUN python -m nltk.downloader -d /data/nltk_data/ stopwords
RUN python -m nltk.downloader -d /data/nltk_data/ punkt
RUN python -m nltk.downloader -d /data/nltk_data/ wordnet
RUN python -m nltk.downloader -d /data/nltk_data/ averaged_perceptron_tagger

EXPOSE 6543
WORKDIR /app

# HEALTHCHECK --interval=1m --timeout=5s --start-period=1m \
#   CMD nc -z -w5 127.0.0.1 6543 || exit 1
# ENV PATH /data/.local/bin:/bin:/usr/bin:/usr/local/bin
# USER nlp
ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]
# RUN /app/docker/docker-setup.sh
CMD ["start"]

# CMD ["bash"]
