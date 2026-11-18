import os

import drive_pipeline as dp


def setup(tmp_path, monkeypatch):
    src = tmp_path / "drive" / "narration.wav"
    src.parent.mkdir()
    src.write_bytes(b"raw audio")
    out = tmp_path / "drive" / "output"
    out.mkdir()
    calls = []

    def fake_normalize(raw, dst, lufs, tp, **kw):
        calls.append((lufs, tp))
        dst.write_bytes(b"normalized " + raw.read_bytes())

    monkeypatch.setattr(dp.normalize_audio, "normalize", fake_normalize)
    return src, out, tmp_path / "work" / "input", calls


def test_normalizes_once_then_reuses(tmp_path, monkeypatch):
    src, out, work_in, calls = setup(tmp_path, monkeypatch)
    first = dp.normalized_narration(src, out, work_in, -14, -1.5)
    assert len(calls) == 1
    assert first.read_bytes() == b"normalized raw audio"
    assert (out / "narration_norm.wav").read_bytes() == b"normalized raw audio"

    again = dp.normalized_narration(src, out, work_in, -14, -1.5)  # same session: local copy
    assert len(calls) == 1 and again == first


def test_new_session_reuses_drive_copy(tmp_path, monkeypatch):
    src, out, work_in, calls = setup(tmp_path, monkeypatch)
    dp.normalized_narration(src, out, work_in, -14, -1.5)
    fresh = tmp_path / "fresh" / "input"  # a new Colab session has an empty work dir
    got = dp.normalized_narration(src, out, fresh, -14, -1.5)
    assert len(calls) == 1
    assert got.parent == fresh and got.read_bytes() == b"normalized raw audio"


def test_changed_narration_or_settings_renormalize(tmp_path, monkeypatch):
    src, out, work_in, calls = setup(tmp_path, monkeypatch)
    dp.normalized_narration(src, out, work_in, -14, -1.5)

    dp.normalized_narration(src, out, work_in, -16, -1.5)  # different target
    assert calls[-1] == (-16, -1.5) and len(calls) == 2

    src.write_bytes(b"new narration take")  # replaced narration.wav
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    got = dp.normalized_narration(src, out, work_in, -16, -1.5)
    assert len(calls) == 3 and got.read_bytes() == b"normalized new narration take"


def test_incomplete_drive_copy_is_not_trusted(tmp_path, monkeypatch):
    src, out, work_in, calls = setup(tmp_path, monkeypatch)
    dp.normalized_narration(src, out, work_in, -14, -1.5)
    (out / dp.NORM_META).unlink()  # e.g. the previous run stopped before finishing the copy
    dp.normalized_narration(src, out, tmp_path / "fresh", -14, -1.5)
    assert len(calls) == 2


def test_music_mix_reused_until_something_changes(tmp_path, monkeypatch):
    narration = tmp_path / "work" / "narration_norm.wav"
    narration.parent.mkdir()
    narration.write_bytes(b"narration")
    music = tmp_path / "drive" / "528Hz_ambient.m4a"
    music.parent.mkdir()
    music.write_bytes(b"music")
    calls = []

    def fake_mix(n, m, out, level, fade, n_lufs, tp):
        calls.append(level)
        out.write_bytes(b"mix")
        return out

    monkeypatch.setattr(dp.background_music, "mix", fake_mix)
    work_in = tmp_path / "work"
    out = dp.mixed_narration(narration, music, work_in, -20, 8, -14, -1.5)
    assert out.name == "narration_mix.wav" and len(calls) == 1
    dp.mixed_narration(narration, music, work_in, -20, 8, -14, -1.5)
    assert len(calls) == 1  # reused
    dp.mixed_narration(narration, music, work_in, -24, 8, -14, -1.5)
    assert calls[-1] == -24 and len(calls) == 2  # level changed
