import json
import shutil
import subprocess

import pytest

import normalize_audio

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def make_tone(path, amplitude, rate=44100, channels=1):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d=4:sample_rate={rate}",
         "-af", f"volume={amplitude}", "-ac", str(channels), str(path)],
        check=True,
    )


def stream_info(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["streams"][0]


def test_quiet_mono_reaches_target(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    make_tone(src, 0.01)
    report = normalize_audio.normalize(src, dst)
    assert abs(float(report["output_i"]) - normalize_audio.TARGET_I) < 0.5
    assert float(report["output_tp"]) <= normalize_audio.TARGET_TP + 0.1
    info = stream_info(dst)
    assert info == {"codec_name": "pcm_s24le", "sample_rate": "48000", "channels": 1}


def test_stereo_layout_kept_and_mono_flag(tmp_path):
    src = tmp_path / "in.wav"
    make_tone(src, 0.5, channels=2)
    normalize_audio.normalize(src, tmp_path / "keep.wav")
    assert stream_info(tmp_path / "keep.wav")["channels"] == 2
    normalize_audio.normalize(src, tmp_path / "mono.wav", mono=True)
    assert stream_info(tmp_path / "mono.wav")["channels"] == 1
