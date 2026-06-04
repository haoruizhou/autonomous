from pathlib import Path

import pytest

from pipeline_v2 import sfm


def _make_project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "images").mkdir(parents=True)
    (proj / "config.yaml").write_text("processes: 2\n")
    return proj


def test_run_opensfm_local_runs_bin_opensfm_per_stage_without_docker(tmp_path, monkeypatch):
    proj = _make_project(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(sfm.subprocess, "run", fake_run)

    sfm.run_opensfm_local(proj, stages=("detect_features", "reconstruct"),
                          opensfm_dir="/source/OpenSfM")

    assert len(calls) == 2
    for (cmd, kwargs), stage in zip(calls, ("detect_features", "reconstruct")):
        assert cmd[0].endswith("bin/opensfm")
        assert cmd[1] == stage
        assert cmd[2] == str(proj.resolve())  # run_opensfm_local resolves the path
        assert kwargs.get("cwd") == "/source/OpenSfM"
        assert "docker" not in cmd


def test_run_opensfm_local_requires_images_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm.subprocess, "run", lambda *a, **k: None)
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        sfm.run_opensfm_local(missing)


def test_run_opensfm_local_requires_config_when_images_present(tmp_path, monkeypatch):
    monkeypatch.setattr(sfm.subprocess, "run", lambda *a, **k: None)
    proj = tmp_path / "noconfig"
    (proj / "images").mkdir(parents=True)  # images present, config.yaml absent
    with pytest.raises(FileNotFoundError):
        sfm.run_opensfm_local(proj)


def test_run_sfm_auto_uses_local_when_in_container(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(sfm, "run_opensfm_local", lambda *a, **k: called.setdefault("local", True))
    monkeypatch.setattr(sfm, "run_opensfm", lambda *a, **k: called.setdefault("docker", True))
    monkeypatch.setenv("AUTONOMOUS_IN_CONTAINER", "1")
    sfm.run_sfm(tmp_path, mode="auto")
    assert called == {"local": True}


def test_run_sfm_auto_uses_docker_when_not_in_container(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(sfm, "run_opensfm_local", lambda *a, **k: called.setdefault("local", True))
    monkeypatch.setattr(sfm, "run_opensfm", lambda *a, **k: called.setdefault("docker", True))
    monkeypatch.delenv("AUTONOMOUS_IN_CONTAINER", raising=False)
    sfm.run_sfm(tmp_path, mode="auto")
    assert called == {"docker": True}


def test_run_sfm_explicit_mode_overrides_env(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(sfm, "run_opensfm_local", lambda *a, **k: called.setdefault("local", True))
    monkeypatch.setattr(sfm, "run_opensfm", lambda *a, **k: called.setdefault("docker", True))
    monkeypatch.setenv("AUTONOMOUS_IN_CONTAINER", "1")
    sfm.run_sfm(tmp_path, mode="docker")
    assert called == {"docker": True}
