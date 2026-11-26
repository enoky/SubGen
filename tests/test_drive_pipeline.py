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

    def fake_mix(n, m, out, level, fade, n_lufs, tp, offset=0.0, fade_in=0.0):
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
    dp.mixed_narration(narration, music, work_in, -24, 2, -14, -1.5, offset=10, fade_in=2)
    assert len(calls) == 3  # Shorts settings (offset, fades) changed


def test_colab_detection(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_2026")
    assert dp.on_colab() and dp.default_root() == dp.DEFAULT_ROOT
    assert dp.default_work_dir(tmp_path) == dp.Path("/content/subgen-work")


def test_local_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.setattr(dp, "on_colab", lambda: False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    project = tmp_path / "videos" / "subgen_shorts"
    project.mkdir(parents=True)

    monkeypatch.chdir(project)
    assert dp.default_root() is None  # no narration here yet
    (project / "narration.wav").write_bytes(b"")
    assert dp.default_root() == project

    work = dp.default_work_dir(project)
    assert work.parent == tmp_path / "appdata" / "SubGen" / "work"
    assert work.name.startswith("subgen_shorts-")
    assert dp.default_work_dir(project) == work  # stable between runs
    assert dp.default_work_dir(tmp_path / "videos" / "other") != work


def test_music_found_in_the_main_folder_next_to_a_shorts_folder(tmp_path):
    shorts = tmp_path / "subgen_shorts"
    cands = dp.music_candidates(shorts, "528Hz_ambient.m4a")
    assert cands[0] == shorts / "528Hz_ambient.m4a"
    assert cands[1] == tmp_path / "subgen" / "528Hz_ambient.m4a"
    main = dp.music_candidates(tmp_path / "subgen", "m.m4a")  # already the main folder: no duplicate
    assert main == [tmp_path / "subgen" / "m.m4a", dp.DEFAULT_ROOT / "m.m4a"]
