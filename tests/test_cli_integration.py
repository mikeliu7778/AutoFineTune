from pathlib import Path

from typer.testing import CliRunner

from autofinetune.cli import app


def test_cli_run_with_fake_trainer(tmp_path: Path, monkeypatch):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "brief.md").write_text("Domain: warehouse robots safety rules", encoding="utf-8")
    runs = tmp_path / "runs"
    # Force fake LLM via env understood by CLI
    monkeypatch.setenv("AUTOFINETUNE_LLM", "fake")
    monkeypatch.setenv("AUTOFINETUNE_TRAINER", "fake")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", str(inp), "--runs-dir", str(runs), "--base-model", "auto"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert any(runs.iterdir())
