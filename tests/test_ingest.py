from pathlib import Path

import pytest

from autofinetune.config import load_config
from autofinetune.errors import FatalError
from autofinetune.ingest.bundle import ingest_bundle
from autofinetune.schemas import DataRoute


def test_brief_only_routes_none(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("Internal wiki about ACME billing", encoding="utf-8")
    cfg = load_config(None)
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.none
    assert "ACME" in result.brief


def test_empty_input_fatal(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    with pytest.raises(FatalError):
        ingest_bundle(inp, cfg)


def test_full_qa_route(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    lines = [
        '{"question":"Q%d","answer":"A%d"}' % (i, i)
        for i in range(cfg.data.min_qa_for_full)
    ]
    (inp / "qa.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.full
    assert len(result.qa) == cfg.data.min_qa_for_full


def test_partial_qa_route(tmp_path: Path):
    inp = tmp_path / "in"
    inp.mkdir()
    cfg = load_config(None)
    n = max(1, cfg.data.min_qa_for_full // 5)
    lines = ['{"question":"Q%d","answer":"A%d"}' % (i, i) for i in range(n)]
    (inp / "qa.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (inp / "brief.md").write_text("domain", encoding="utf-8")
    result = ingest_bundle(inp, cfg)
    assert result.route == DataRoute.partial
