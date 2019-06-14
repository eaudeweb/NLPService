from click.testing import CliRunner


def test_it(tmpdir):
    from nlpservice.nlp.fasttext import main
    from pkg_resources import resource_filename

    textfile = resource_filename('nlpservice', 'tests/fixtures/corpus.txt')
    output = tmpdir / 'test-corpus-fasttext'

    runner = CliRunner()
    result = runner.invoke(main, [textfile, str(output)])
    assert result.exit_code == 0

    from gensim.models import FastText
    model = FastText.load(str(output))

    vocab = model.wv.vocab
    assert 'trojan' in vocab
    assert model['trojan'].shape == (100,)
