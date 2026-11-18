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


def timed(start, end, fallback=(0.0, 60.0)):
    return subgen.TimedWord("w", start, end, fallback[0], fallback[1], 0, 1)


def test_script_sentence_times_ignore_stray_words():
    # an unaligned word (e.g. a number) must not pull in its 0-60 s Whisper segment
    assert subgen.script_sentence_times(
        [(timed(10.0, 10.4), 0.0), (timed(None, None), 1.0), (timed(11.0, 12.5), 2.0)]
    ) == (10.0, 12.5)
    # a transcript-only word (.5 position) at the next sentence's boundary is ignored
    assert subgen.script_sentence_times([(timed(10.0, 10.4), 0.0), (timed(13.8, 14.1), 0.5)]) == (10.0, 10.4)
    # segment bounds only as a last resort
    assert subgen.script_sentence_times([(timed(None, None, (20.0, 25.0)), 0.0)]) == (20.0, 25.0)
