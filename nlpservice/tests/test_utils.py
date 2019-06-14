
class TestUtils:

    def test_lemmatize(self, kg):
        from nlpservice.nlp.utils import lemmatize_kg_terms
        terms = lemmatize_kg_terms(kg['Threats'])
        assert len(terms) == 99

        assert ('softwar', 'vulner') in terms
        assert ('zero-day',) in terms
        assert ('man-in-the-brows',) in terms
