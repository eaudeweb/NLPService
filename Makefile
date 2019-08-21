.DEFAULT_GOAL := all

MINICONDA = Miniconda3-latest-Linux-x86_64.sh

$(MINICONDA):
	wget "https://repo.anaconda.com/miniconda/$(MINICONDA)"

.PHONY: bootstrap
bootstrap: $(MINICONDA)
	@set -x; \
	sh $(MINICONDA) -u; \
	sh -c "conda create -n nlpservice python=3.7"; \
	sh -c "conda activate nlpservice"; \
	sh -c "conda install regex;" \
	sh -c "pip install allennlp"; \
	sh -c "conda install -c anaconda tensorflow-gpu"
	pip install -r requirements.txt
	pip install -e .

all:	bootstrap
