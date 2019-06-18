class TestKGParsing:

    def test_major_branches(self, kg):
        assert len(kg) == 9
        assert 'Assets' in kg

    def test_flattened_branch(self, kg):
        assert 'Trojan' in kg['Threats']
        assert 'Botnets' in kg['Threats']
        assert 'Hacktivists' in kg['Threat actors']

    def test_major_branch_is_trigger(self, kg):
        for mj in kg:
            assert mj in kg[mj]


class TestClassify:

    def test_read_corpus(self):
        from pkg_resources import resource_filename
        from nlpservice.nlp.classify import read_corpus

        fpath = resource_filename('nlpservice', 'tests/fixtures/corpus.txt')
        docs = read_corpus(fpath)

        assert len(docs) == 15
        assert all([len(d) > 0 for d in docs])

        assert docs[0][0] == 'qadars trojan returns bigger badder ever'
        assert len(docs[0]) == 6
        assert docs[0][-1] == 'view full story original source softpedia'

    def test_stream_corpus(self):
        from pkg_resources import resource_filename
        from nlpservice.nlp.classify import stream_corpus

        fpath = resource_filename('nlpservice', 'tests/fixtures/corpus.txt')
        docs = stream_corpus(fpath)

        doc = next(docs)

        assert doc[0] == 'qadars trojan returns bigger badder ever'
        assert len(doc) == 6
        assert doc[-1] == 'view full story original source softpedia'

    def test_get_doc_labels(self, corpus, lemmatized_kg):
        from nlpservice.nlp.classify import get_doc_labels

        doc = corpus[0]
        labels = get_doc_labels(doc, lemmatized_kg)
        assert labels == 'Threats'

    def test_prepare_corpus(self, corpus, lemmatized_kg):
        from nlpservice.nlp.classify import prepare_corpus

        X, y = prepare_corpus(corpus, lemmatized_kg)

        assert len(X) == len(y) == 15
        assert y[0] == 'Threats'
        assert X[0] is corpus[0]

    def test_docs_to_dtm(self, corpus):
        from gensim.corpora import Dictionary
        from nlpservice.nlp.classify import docs_to_dtm

        sents = []

        for doc in corpus:
            for line in doc:
                tokens = line.split(' ')
                sents.append(tokens)

        dct = Dictionary(sents)

        X = docs_to_dtm(corpus, dct.token2id, 300)
        assert X.shape == (15, 300)     # 15 documents, 300 columns

        t = [7,  9,  8,  5,  4,  6,  1, 15, 19, 13,  7, 11,  9]
        assert list(X[0][:len(t)]) == t

    def test_make_classifier(self,
                             ftmodel, corpus, lemmatized_kg, tf_session):
        from nlpservice.nlp.classify import make_classifier
        model = make_classifier(ftmodel, corpus, lemmatized_kg)
        assert model

    def test_predict(self,
                     k_model, ftmodel, corpus, lemmatized_kg):

        from nlpservice.nlp.classify import docs_to_dtm
        X = docs_to_dtm([corpus[0]], ftmodel.wv.vocab, 300)

        assert k_model.predict(X).shape == (1, 9)

    def test_create_model(self, tf_session):
        import numpy as np
        from nlpservice.nlp.classify import create_model
        model = create_model(1000, 100, 100, np.empty((1000, 100)), 2)
        assert model.summary()
