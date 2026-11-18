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
