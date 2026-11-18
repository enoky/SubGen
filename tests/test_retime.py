import shutil
import subprocess

import pytest
import srt

import retime as r

# Speech at 1.0-3.0 (with a comma pause 1.8-2.0), 4.0-6.0 and 7.0-9.0; pauses in between.
PAUSES = [(0.0, 1.0), (1.8, 2.0), (3.0, 4.0), (6.0, 7.0), (9.0, 12.0)]
TRUE = [(1.0, 3.0), (4.0, 6.0), (7.0, 9.0)]
WANT = [(0.95, 3.3), (3.95, 6.3), (6.95, 9.3)]


def approx(times):
    return [pytest.approx(t, abs=1e-3) for t in times]


def test_correct_times_only_get_padding():
    assert r.retime(TRUE, PAUSES) == approx(WANT)


def test_edges_inside_pauses_snap_to_speech():
    bad = [(0.4, 3.0), (4.0, 6.7), (6.2, 9.8)]  # early start, late end, both
    assert r.retime(bad, PAUSES) == approx(WANT)


def test_overlapping_neighbours_split_at_the_sentence_gap():
    bad = [(1.0, 4.4), (4.0, 6.0), (5.6, 9.0)]  # end runs into the next sentence; start inside the previous one
    assert r.retime(bad, PAUSES) == approx(WANT)


def test_comma_pause_is_not_a_sentence_break():
    assert r.retime([(1.0, 3.0)], PAUSES) == approx([(0.95, 3.3)])
    assert r.retime([(1.9, 3.0)], PAUSES)[0][0] == pytest.approx(1.95)  # a start in the comma pause snaps forward only


def test_retiming_twice_changes_nothing():
    once = r.retime([(0.4, 3.6), (3.3, 6.8), (6.2, 9.8)], PAUSES)
    assert r.retime(once, PAUSES) == approx(once)


def test_padding_never_reaches_the_next_cue():
    tight = [(1.0, 3.0), (3.1, 6.0)]
    pauses = [(0.0, 1.0), (3.0, 3.1), (6.0, 9.0)]
    (s1, e1), (s2, e2) = r.retime(tight, pauses)
    assert e1 < s2 and s2 >= 3.0


def test_report_lists_big_moves():
    new, changes = r.retime_with_report([(0.4, 3.0), (4.0, 6.0)], PAUSES)
    assert [c.index for c in changes] == [0, 1]  # 0: start moved; 1: only its 0.3 s tail (> 0.15)
    assert "adjusted 2 of 2" in r.summarize(changes, 2, ["a", "b"])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_retime_srt_on_real_audio(tmp_path):
    audio = tmp_path / "n.wav"
    # tone (speech stand-in) at 1-3 s and 4-6 s, silence elsewhere, 8 s total
    expr = "if(between(t,1,3)+between(t,4,6),0.5*sin(2*PI*220*t),0)"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=16000:d=8", str(audio)], check=True)
    subs = tmp_path / "n.srt"
    subs.write_text(srt.compose([
        srt.Subtitle(1, srt.timedelta(seconds=0.3), srt.timedelta(seconds=3.0), "One."),
        srt.Subtitle(2, srt.timedelta(seconds=4.0), srt.timedelta(seconds=7.2), "Two."),
    ]), encoding="utf-8")
    r.retime_srt(audio, subs)
    got = [(s.start.total_seconds(), s.end.total_seconds()) for s in srt.parse(subs.read_text())]
    assert got[0][0] == pytest.approx(0.95, abs=0.03) and got[0][1] == pytest.approx(3.3, abs=0.03)
    assert got[1][0] == pytest.approx(3.95, abs=0.03) and got[1][1] == pytest.approx(6.3, abs=0.03)


# Subtitle 258/259 from a real video: 258 opens with "In its words," and a 1 s
# dramatic pause before a long quote; 259 starts with "Reason", a word inside that
# quote. 258 ended up cut at the dramatic pause and 259 took the rest of its time.
QUOTE = 'In its words, "But, the report of Reason must be hospitably received, and treated with respect."'
NEXT = "Reason can't lift the veil of the unknowable, the Initiates said, but it can tell us certain things."
CASE_PAUSES = [(0.0, 0.5), (1.35, 2.35), (2.65, 2.86), (5.77, 5.96), (7.47, 7.97),
               (10.61, 10.82), (11.98, 12.2), (16.24, 17.5)]
CASE_WANT = [(0.45, 7.77), (7.92, 16.54)]


def test_subtitle_cut_at_a_pause_inside_its_sentence_is_repaired():
    broken = [(0.45, 1.65), (2.30, 16.54)]  # 258 shown for 1.2 s at ~70 letters/s
    assert r.retime(broken, CASE_PAUSES, [QUOTE, NEXT]) == approx(CASE_WANT)


def test_overlap_is_split_where_the_text_lengths_say_not_at_the_longest_pause():
    stolen = [(0.5, 7.47), (3.9, 16.24)]  # 259 timed from "Reason" inside the quote
    assert r.retime(stolen, CASE_PAUSES, [QUOTE, NEXT]) == approx(CASE_WANT)


def test_normal_pace_is_left_alone():
    assert r.retime(CASE_WANT, CASE_PAUSES, [QUOTE, NEXT]) == approx(CASE_WANT)
    assert r.retime(CASE_WANT, CASE_PAUSES, [QUOTE, NEXT]) == r.retime(CASE_WANT, CASE_PAUSES)
