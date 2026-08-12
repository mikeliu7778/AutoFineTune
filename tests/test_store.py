from pathlib import Path

from autofinetune.schemas import BaseModelChoice, RunStatus
from autofinetune.store.run_store import RunStore


def test_create_and_reload_run(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("domain", encoding="utf-8")
    rec = store.create(input_dir=inp)
    assert rec.run_id
    assert rec.status == RunStatus.created
    loaded = store.load(rec.run_id)
    assert loaded.run_id == rec.run_id
    assert (store.root / rec.run_id / "run.json").is_file()
    assert (store.root / rec.run_id / "input" / "brief.md").is_file()


def test_pause_flag_round_trip(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("x", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    store.request_pause(rec.run_id)
    loaded = store.load(rec.run_id)
    assert loaded.pause_requested is True


def test_set_base_model_persisted(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("x", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    choice = BaseModelChoice(model_id="Qwen/Qwen2.5-7B-Instruct", mode="user", rationale="pin")
    store.set_base_model(rec.run_id, choice)
    loaded = store.load(rec.run_id)
    assert loaded.base_model is not None
    assert loaded.base_model.model_id.endswith("7B-Instruct")


def test_trainer_backend_persisted(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "brief.md").write_text("x", encoding="utf-8")
    rec = store.create(input_dir=tmp_path / "in")
    store.set_trainer_backend(rec.run_id, "trl")
    loaded = store.load(rec.run_id)
    assert loaded.trainer_backend == "trl"
