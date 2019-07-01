prepare --es-url=http://localhost:9200/content data/corpus.txt
label data/corpus.txt data/labeled-corpus --kg-url=http://localhost:8880/api/knowledge-graph/dump_all/
docker run --rm -v /home/tibi/work/enisa-opencsam/NLPService/data:/data -it hephaex/fasttext bash
./fasttext supervised -input /data/labeled-corpus-train -output /data/labeled-corpus -lr 0.5 -epoch 40 -wordNgrams 2 -bucket 2000000 -dim 50
./fasttext test /data/labeled-corpus.bin /data/labeled-corpus-test 3
