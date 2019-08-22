.DEFAULT_GOAL := help

MINICONDA = Miniconda3-latest-Linux-x86_64.sh

$(MINICONDA):
	wget "https://repo.anaconda.com/miniconda/$(MINICONDA)"

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

.PHONY: release¬
release-frontend:		## Make a Docker Hub release for nlpservice
	sh -c "cd frontend && docker build -t tiberiuichim/nlpservice:$(VERSION) -f Dockerfile . && docker push tiberiuichim/nlpservice:$(VERSION)"

.PHONY: help
help:		## Show this help.
	@echo -e "$$(grep -hE '^\S+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##\s*/:/' -e 's/^\(.\+\):\(.*\)/\\x1b[36m\1\\x1b[m:\2/' | column -c2 -t -s :)"
