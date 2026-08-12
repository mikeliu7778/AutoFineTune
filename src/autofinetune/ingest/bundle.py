from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from autofinetune.config import AppConfig
from autofinetune.errors import FatalError
from autofinetune.schemas import DataRoute, QAItem

_DOC_SUFFIXES = {".md", ".txt", ".markdown"}


class IngestResult(BaseModel):
    route: DataRoute
    brief: str = ""
    docs_text: str = ""
    qa: list[QAItem] = Field(default_factory=list)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _load_docs(docs_dir: Path) -> str:
    if not docs_dir.is_dir():
        return ""
    chunks: list[str] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in _DOC_SUFFIXES:
            chunks.append(f"# {path.name}\n{_read_text(path)}")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise FatalError("pypdf required to read PDF docs; pip install pypdf") from e
            reader = PdfReader(str(path))
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
            if text:
                chunks.append(f"# {path.name}\n{text}")
    return "\n\n".join(chunks).strip()


def _load_qa(path: Path) -> list[QAItem]:
    if not path.is_file():
        return []
    items: list[QAItem] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            items.append(QAItem.model_validate(raw))
        except Exception as e:
            raise FatalError(f"invalid qa.jsonl line {line_no}: {e}") from e
    return items


def ingest_bundle(input_dir: Path, cfg: AppConfig) -> IngestResult:
    brief_path = input_dir / "brief.md"
    brief = _read_text(brief_path) if brief_path.is_file() else ""
    docs_text = _load_docs(input_dir / "docs")
    qa = _load_qa(input_dir / "qa.jsonl")

    if not brief and not docs_text and not qa:
        raise FatalError(
            "Minimum input required: non-empty brief.md, docs/, or qa.jsonl"
        )

    if not qa:
        route = DataRoute.none
    elif len(qa) >= cfg.data.min_qa_for_full:
        route = DataRoute.full
    else:
        route = DataRoute.partial

    return IngestResult(route=route, brief=brief, docs_text=docs_text, qa=qa)
