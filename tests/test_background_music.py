import re
import shutil
import subprocess
from pathlib import Path

import pytest

import background_music as bm


def test_music_gain_is_relative_to_narration():
    assert bm.music_gain(music_lufs=-16.0, narration_lufs=-14.0, level=-20.0) == -18.0
    assert bm.music_gain(music_lufs=-40.0, narration_lufs=-14.0, level=-20.0) == 6.0  # quiet tracks go up


def test_mix_command_trims_fades_and_limits():
    cmd = bm.build_mix_command(Path("n.wav"), Path("m.m4a"), Path("o.wav"), 6102.13, -18.0, fade=8)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "atrim=0:6102.130000" in graph
    assert "volume=-18.00dB" in graph
    assert "afade=t=out:st=6094.130000:d=8.000000" in graph
    assert "amix=inputs=2:duration=first:normalize=0" in graph
    assert "alimiter=limit=0.8414:level=false" in graph  # -1.5 dBTP
    assert "-stream_loop" not in cmd
    assert "-stream_loop" in bm.build_mix_command(Path("n"), Path("m"), Path("o"), 60, 0, loop=True)


def test_fade_never_longer_than_the_video():
    graph = bm.build_mix_command(Path("n"), Path("m"), Path("o"), 5.0, 0, fade=8)
    graph = graph[graph.index("-filter_complex") + 1]
    assert "afade=t=out:st=0.000000:d=5.000000" in graph


def lufs(path, ss, t):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(ss), "-t", str(t), "-i", str(path),
                          "-af", "ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True)
    return float(re.findall(r"\bI:\s+(-?[\d.]+) LUFS", out.stderr)[-1])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_mix_sets_music_level_and_fades_out(tmp_path):
    silent, music, out = tmp_path / "n.wav", tmp_path / "m.m4a", tmp_path / "mix.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "20",
                    str(silent)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:a=0.3",
                    "-ac", "2", "-c:a", "aac", str(music)], check=True)
    bm.mix(silent, music, out, level=-20, fade=4, narration_lufs=-14.0)
    assert abs(lufs(out, 2, 12) - (-34.0)) < 1.0  # the music alone, 20 LU under a -14 LUFS narration
    assert lufs(out, 18, 2) < -40  # fading out
