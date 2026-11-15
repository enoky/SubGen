import shutil
import subprocess

import pytest

from ffmpeg_progress import run_ffmpeg

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_success_returns_stderr(capsys):
    cmd = ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "sine=d=3", "-af", "volumedetect", "-f", "null", "-"]
    stderr = run_ffmpeg(cmd, 3.0, "test")
    assert "mean_volume" in stderr
    assert "100%" in capsys.readouterr().err  # bar reached the end


def test_failure_raises_with_stderr():
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_ffmpeg(["ffmpeg", "-i", "/nonexistent.wav", "-f", "null", "-"], 1.0, "test")
    assert "nonexistent" in exc.value.stderr


def test_unknown_duration():
    run_ffmpeg(["ffmpeg", "-f", "lavfi", "-i", "sine=d=1", "-f", "null", "-"], None, "test")
