import pytest

import phrases as p


def test_split_prefers_punctuation_and_keeps_phrases_short():
    text = ("Reason can't lift the veil of the unknowable, the Initiates said, "
            "but it can tell us certain things the mind is compelled to accept.")
    parts = p.split_phrases(text)
    assert " ".join(parts) == text
    assert parts[1] == "the veil of the unknowable,"  # runs a little long to end at the comma
    assert "the Initiates said," in parts
    assert all(len(x) <= p.MAX_CHARS + p.OVERFLOW for x in parts)


def test_no_phrase_ends_on_a_weak_word():
    for text in ["The Kybalion answers that question with a gentle and rather surprising permission.",
                 "Philosophy asked after what could be known, while metaphysics tried to cross over into the unknowable."]:
        for part in p.split_phrases(text)[:-1]:
            assert p.word_key(part.split()[-1]) not in p.WEAK_ENDINGS or p.BREAK_AFTER.search(part)


def test_short_sentence_stays_whole_and_no_lone_last_word():
    assert p.split_phrases("Stay with me until the end.") == ["Stay with me until the end."]
    assert p.split_phrases("Thank you for listening to this whole episode, friends.")[-1] != "friends."


CUE = (10.0, 16.0, "Reason can't lift the veil of the unknowable, the Initiates said.")
# speech 10.0-13.9, a pause after "unknowable," until 14.2, speech 14.2-16.0
PAUSES = [(9.0, 10.0), (13.9, 14.2), (16.0, 17.0)]


def test_phrases_follow_the_speech_and_snap_to_pauses():
    out = p.phrase_cues([CUE], PAUSES)
    assert [x.text for x in out] == ["Reason can't lift", "the veil of the unknowable,", "the Initiates said."]
    assert out[0].start == 10.0 and out[-1].end == 16.0
    assert out[2].start == pytest.approx(14.2)  # after the comma: when speech resumes
    assert all(a.end == b.start for a, b in zip(out, out[1:]))  # no flicker between phrases


def test_word_timings_win_even_with_a_misheard_word():
    words = [["Reason", 10.0, 10.4], ["can't", 10.4, 10.7], ["lift", 10.7, 11.0], ["the", 11.05, 11.15],
             ["veal", 11.15, 11.5], ["of", 11.5, 11.6], ["the", 11.6, 11.7], ["unknowable", 11.7, 13.9],
             ["the", 14.25, 14.4], ["Initiates", 14.4, 15.2], ["said", 15.2, 15.8]]
    out = p.phrase_cues([CUE], PAUSES, words)
    assert out[1].start == pytest.approx(11.05)  # "the" of "the veil", from its word timing
    assert out[2].start == pytest.approx(14.25)  # word timing beats the pause estimate


def test_word_far_outside_its_cue_is_ignored():
    words = [["the", 40.0, 40.2], ["Initiates", 40.2, 40.9]]  # a match from somewhere else entirely
    out = p.phrase_cues([CUE], PAUSES, words)
    assert 10.0 < out[2].start < 16.0


def test_too_short_to_split_stays_one_subtitle():
    out = p.phrase_cues([(5.0, 5.5, "Reason can't lift the veil of the unknowable, the Initiates said.")], [])
    assert len(out) == 1 and out[0].text.startswith("Reason")
