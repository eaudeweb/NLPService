from unittest.mock import patch

from click.testing import CliRunner


@patch('nlpservice.nlp.prepare.get_es_records')
def test_it(get_es_records, tmpdir, es_docs):
    from nlpservice.nlp.prepare import main

    get_es_records.return_value = es_docs

    runner = CliRunner()
    output = tmpdir / 'out.txt'
    result = runner.invoke(main, [str(output)])
    assert result.exit_code == 0

    out = output.read()
    assert out.splitlines()[0] == 'qadars trojan returns bigger badder ever'
