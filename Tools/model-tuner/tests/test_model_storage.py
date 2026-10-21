"""Model Tuner finds models where the dashboard loads them: pinned on this PC, the rest on the model server."""

import json
import os
import time

from modeltuner import paths, profile


def _write(path, *, settled=True):
    path.write_bytes(b"x")
    if settled:
        old = time.time() - 3600
        os.utime(path, (old, old))


def _setup(tmp_path, monkeypatch, *, network=True, pinned=()):
    local, server = tmp_path / "local", tmp_path / "server"
    local.mkdir()
    server.mkdir()
    cfg = tmp_path / "model-storage.json"
    cfg.write_text(json.dumps({"networkDir": str(server if network else tmp_path / "offline"),
                               "pinnedLocal": list(pinned)}), encoding="utf-8")
    monkeypatch.setattr(paths, "MODELS_DIR", local)
    monkeypatch.setattr(paths, "MODEL_STORAGE_JSON", cfg)
    return local, server


def test_unpinned_model_loads_from_the_server(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch)
    for d in (local, server):
        _write(d / "m.gguf")
    assert paths.resolve_model("m.gguf") == server / "m.gguf"


def test_pinned_model_loads_from_this_pc(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch, pinned=["m.gguf"])
    for d in (local, server):
        _write(d / "m.gguf")
    assert paths.resolve_model("m.gguf") == local / "m.gguf"


def test_offline_server_falls_back_to_this_pc(tmp_path, monkeypatch):
    local, _ = _setup(tmp_path, monkeypatch, network=False)
    _write(local / "m.gguf")
    assert paths.model_dirs("m.gguf") == [local]
    assert paths.resolve_model("m.gguf") == local / "m.gguf"


def test_missing_or_broken_config_means_local_only(tmp_path, monkeypatch):
    local, _ = _setup(tmp_path, monkeypatch)
    paths.MODEL_STORAGE_JSON.write_text("{ not json", encoding="utf-8")
    assert paths.model_dirs() == [local]
    monkeypatch.setattr(paths, "MODEL_STORAGE_JSON", tmp_path / "absent.json")
    assert paths.model_dirs() == [local]


def test_list_models_merges_both_places_once(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(profile, "read_gguf", lambda p, **_: seen.append(p) or {"general.type": "model"})
    for p in (local / "a.gguf", local / "b.gguf", server / "b.gguf", server / "c.gguf"):
        _write(p)
    assert profile.list_models() == ["a.gguf", "b.gguf", "c.gguf"]
    assert server / "b.gguf" in seen and local / "b.gguf" not in seen


def test_pins_ignore_case_like_windows(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch, pinned=["M.GGUF"])
    for d in (local, server):
        _write(d / "m.gguf")
    assert paths.resolve_model("m.gguf") == local / "m.gguf"


def test_listing_reads_the_pinned_local_copy(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch, pinned=["m.gguf"])
    seen = []
    monkeypatch.setattr(profile, "read_gguf", lambda p, **_: seen.append(p) or {"general.type": "model"})
    for d in (local, server):
        _write(d / "m.gguf")
    assert profile.list_models() == ["m.gguf"]
    assert seen == [local / "m.gguf"]


def test_server_copy_still_being_written_is_skipped(tmp_path, monkeypatch):
    local, server = _setup(tmp_path, monkeypatch)
    _write(local / "m.gguf")
    _write(server / "m.gguf", settled=False)
    _write(server / "new.gguf", settled=False)
    assert paths.resolve_model("m.gguf") == local / "m.gguf"
    assert sorted(paths.model_files()) == ["m.gguf"]
