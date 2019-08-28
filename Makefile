.DEFAULT_GOAL := help

MINICONDA = Miniconda3-latest-Linux-x86_64.sh

$(MINICONDA):
	wget "https://repo.anaconda.com/miniconda/$(MINICONDA)"

# define standard colors
BLACK        := $(shell tput -Txterm setaf 0)
RED          := $(shell tput -Txterm setaf 1)
GREEN        := $(shell tput -Txterm setaf 2)
YELLOW       := $(shell tput -Txterm setaf 3)
LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
PURPLE       := $(shell tput -Txterm setaf 5)
BLUE         := $(shell tput -Txterm setaf 6)
WHITE        := $(shell tput -Txterm setaf 7)

RESET := $(shell tput -Txterm sgr0)

.PHONY: bootstrap
bootstrap: $(MINICONDA)		## Bootstrap for local development
	@echo "Please install miniconda in $(HOME)/miniconda3"
	@set -x; \
	sh $(MINICONDA) -u; \
	source $(HOME)/miniconda3/bin/activate; \
	sh -c "conda create -n nlpservice python=3.7"; \
	sh -c "conda activate nlpservice"; \
	sh -c "conda install regex;" \
	sh -c "pip install allennlp"; \
	sh -c "conda install -c anaconda tensorflow-gpu"
	pip install -r requirements.txt
	pip install -e .
	python -m nltk.downloader stopwords
	python -m nltk.downloader punkt
	python -m nltk.downloader wordnet
	python -m nltk.downloader averaged_perceptron_tagger

.PHONY: release
release:		## Make a Docker Hub release for nlpservice
	sh -c "docker build -t tiberiuichim/nlpservice:$(VERSION) -f Dockerfile . && docker push tiberiuichim/nlpservice:$(VERSION)"

.PHONY: help
help:		## Show this help.
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"

.PHONY: prepare-es
prepare-es:		## Prepare a corpus file reading content from ElasticSearch
	prepare --es-url=http://localhost:9200/content data/corpus.txt

.PHONY: prepare-dump
prepare-dump:		## Prepare a corpus file reading content from an ES dump file
	@echo $(RED)Preparing a corpus file, from ES dump file... $(RESET)
	prepare --input-file=./content.data.json data/corpus.txt

.PHONY: label
label:		## Prepare a FastText compatible labeled corpus file
	@echo $(RED)Creating a fasttext compatible labeled corpus file... $(RESET)
	label data/corpus.txt data/labeled-corpus --kg-url=http://$(API_HOST)/api/knowledge-graph/dump_all/

.PHONY: fasttext
fasttext:		## Make a FastText classifier
	@echo $(RED)Training the Fasttext classifier model... $(RESET)
	docker run --rm -v /home/tibi/work/enisa-opencsam/NLPService/data:/data -it hephaex/fasttext sh -c "
	./fasttext supervised -input /data/labeled-corpus-train -output /data/labeled-corpus -lr 0.5 -epoch 40 -wordNgrams 2 -bucket 2000000 -dim 50;\
	./fasttext test /data/labeled-corpus.bin /data/labeled-corpus-test 3"

.PHONY: wordvectors
wordvectors:		## Create WordVectors model
	@echo $(RED)Training the wordvectors model... $(RESET)
	kv data/corpus.txt data/corpus-ft

.PHONY: train-keras
train-keras:		## Train a Keras classifier
	@echo $(RED)Training Keras classifier model... $(RESET)
	rm -rf $(TMP)/cachedir/*
	train --gpu data/k-model.hdf data/corpus-ft data/corpus.txt --kg-url=$(API_HOST)/api/knowledge-graph/dump_all/

.PHONY: full-train
full-train:	prepare-dump wordvectors train-keras		## Full pipeline to train Keras model
	@echo $(GREEN)Making the Keras Classifier Model $(RESET)

fixtures:		## Make the fixtures needed for automated tests
	prepare --es-url=http://localhost:9200/content nlpservice/tests/fixtures/corpus.txt
	kv nlpservice/tests/fixtures/corpus.txt nlpservice/tests/fixtures/corpus-ft

deploy-models:		## Makes an archive of trained model, to deploy
	@echo $(GREEN)Creating models.tgz archive...$(RESET)
	tar czf model.tgz data/k-model* data/corpus-ft*
