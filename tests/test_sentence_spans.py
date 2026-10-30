import pytest

import subgen


@pytest.fixture(scope="module")
def nlp():
    return subgen.load_sentence_segmenter()


def sentences(nlp, text):
    return [subgen.normalize_text(text[a:b]) for a, b in subgen.sentence_char_spans(nlp, text)]


def test_closing_quote_stays_with_its_sentence(nlp):
    text = (
        'The next sentence reads, "To the chosen ones, it will be completely bestowed."\n\n'
        "He also offers a history of failure."
    )
    assert sentences(nlp, text) == [
        'The next sentence reads, "To the chosen ones, it will be completely bestowed."',
        "He also offers a history of failure.",
    ]


def test_curly_quote_and_bracket(nlp):
    text = "He said, “It is done.”\n\nThen he left. (She stayed.)\n\nThe end."
    out = sentences(nlp, text)
    assert out[0] == "He said, “It is done.”"
    assert not any(s[0] in "”)" for s in out)


def test_opening_quote_is_not_moved(nlp):
    text = 'He paused. "Wisdom is earned," he said.'
    out = sentences(nlp, text)
    assert out[-1].startswith('"Wisdom')
