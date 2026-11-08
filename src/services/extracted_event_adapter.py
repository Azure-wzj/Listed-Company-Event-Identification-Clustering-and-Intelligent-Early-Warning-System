"""结构化抽取结果与系统事件格式之间的转换。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.schemas.event import EventSchema
from src.services.cleaned_xlsx import iter_cleaned_rows
from src.services.event_adapter import EventAdapter


ROOT = Path(__file__).resolve().parents[2]
CLEANED_SOURCE_BY_OUTPUT = {
    "extracted_unlabeled_test": ROOT / "cleaned_data" / "unlabeled_test_cleaned.xlsx",
    "extracted_unlabeled_train": ROOT / "cleaned_data" / "unlabeled_train_cleaned.xlsx",
    "extracted_tagged_test": ROOT / "cleaned_data" / "test_tagged_cleaned.xlsx",
    "extracted_tagged_train": ROOT / "cleaned_data" / "train_tagged_cleaned.xlsx",
}


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


# 显式日期格式，避免 pandas 3.x 的 guess_datetime_format 在批量解析时
# 抛出 SystemError: Unmatched paren in format。
_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%Y-%m",
    "%Y",
    "%Y年%m月%d日",
    "%Y年%m月%d日 %H:%M:%S",
    "%Y年%m月%d日 %H:%M",
)


def _parse_time(value: Any) -> datetime | None:
    text = _safe_str(value)
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    # 兜底：pandas 解析（限定 format，避免格式猜测）
    try:
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    except Exception:
        return None
    return parsed.to_pydatetime() if pd.notna(parsed) else None


def _first_subject(subjects: Any) -> dict[str, Any]:
    if not isinstance(subjects, list):
        return {}
    for item in subjects:
        if isinstance(item, dict) and _safe_str(item.get("trading_code")):
            return item
    for item in subjects:
        if isinstance(item, dict):
            return item
    return {}


def _event_entity(record: dict[str, Any], subject: dict[str, Any]) -> str | None:
    subject_name = _safe_str(subject.get("secu_abbr"))
    if subject_name:
        return subject_name
    impact_objects = record.get("impact_objects")
    if isinstance(impact_objects, list):
        names = [_safe_str(item.get("name")) for item in impact_objects if isinstance(item, dict)]
        names = [name for name in names if name]
        if names:
            return "、".join(names[:3])
    return None


def _fallback_content(record: dict[str, Any], subject: dict[str, Any]) -> str:
    parts = [
        _safe_str(record.get("article_title")),
        _safe_str(record.get("event_name")),
        _safe_str(record.get("impact_direction") or record.get("event_emotion")),
        _safe_str(subject.get("secu_abbr")),
    ]
    industry = record.get("industry")
    if isinstance(industry, dict):
        parts.append(_safe_str(industry.get("induname")))
    impact_objects = record.get("impact_objects")
    if isinstance(impact_objects, list):
        parts.extend(
            _safe_str(item.get("name"))
            for item in impact_objects
            if isinstance(item, dict)
        )
    return " ".join(part for part in parts if part)


class ExtractedEventAdapter:
    """Convert extracted records into EventSchema records."""

    @staticmethod
    def from_record(record: dict[str, Any]) -> EventSchema:
        subject = _first_subject(record.get("subjects"))
        industry = record.get("industry") if isinstance(record.get("industry"), dict) else {}
        content = _safe_str(record.get("content")) or _fallback_content(record, subject)

        return EventSchema(
            article_file_id=_safe_str(record.get("article_file_id")) or "",
            article_title=_safe_str(record.get("article_title")) or "",
            article_publish_time=_parse_time(
                record.get("article_publish_time") or record.get("publish_time")
            ),
            article_source=_safe_str(record.get("article_source")) or "",
            content=content,
            trading_code=_safe_str(subject.get("trading_code")),
            secu_abbr=_safe_str(subject.get("secu_abbr")),
            industry_code=_safe_str(industry.get("inducode") or industry.get("industry_code")),
            industry_name=_safe_str(industry.get("induname") or industry.get("industry_name")),
            event_name=_safe_str(record.get("event_name")),
            event_time=_parse_time(record.get("event_time")),
            event_entity=_event_entity(record, subject),
            event_impact_direction=_safe_str(
                record.get("impact_direction") or record.get("event_emotion")
            ),
        )

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> list[EventSchema]:
        return [cls.from_record(record) for record in records]

    @staticmethod
    def load_records(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("events", [])
        return records

    @staticmethod
    def infer_cleaned_source(path: str | Path) -> Path | None:
        """Find the workbook corresponding to an extraction output."""
        return CLEANED_SOURCE_BY_OUTPUT.get(Path(path).stem)

    @classmethod
    def enrich_records(
        cls,
        records: list[dict[str, Any]],
        cleaned_source: str | Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Fill missing article fields from the source workbook."""
        source = Path(cleaned_source)
        wanted = {
            _safe_str(record.get("article_file_id"))
            for record in records
            if _safe_str(record.get("article_file_id"))
            and any(
                not _safe_str(record.get(field))
                for field in ("article_title", "article_source", "content", "article_publish_time")
            )
        }
        if not wanted or not source.exists():
            return records, {"requested": len(wanted), "matched": 0, "content_restored": 0}

        source_rows: dict[str, dict[str, Any]] = {}
        for row in iter_cleaned_rows(source):
            article_id = _safe_str(row.get("article_file_id"))
            if article_id in wanted:
                source_rows[article_id] = row
                if len(source_rows) == len(wanted):
                    break

        enriched: list[dict[str, Any]] = []
        content_restored = 0
        for original in records:
            record = dict(original)
            article_id = _safe_str(record.get("article_file_id"))
            source_row = source_rows.get(article_id or "")
            if source_row:
                had_content = bool(_safe_str(record.get("content")))
                for field in ("article_title", "article_source", "content", "article_publish_time"):
                    if not _safe_str(record.get(field)) and _safe_str(source_row.get(field)):
                        record[field] = source_row[field]
                if not had_content and _safe_str(record.get("content")):
                    content_restored += 1
            enriched.append(record)
        return enriched, {
            "requested": len(wanted),
            "matched": len(source_rows),
            "content_restored": content_restored,
        }

    @classmethod
    def load_enriched_records(
        cls,
        path: str | Path,
        cleaned_source: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        records = cls.load_records(path)
        source = Path(cleaned_source) if cleaned_source else cls.infer_cleaned_source(path)
        if source is None:
            return records, {"requested": 0, "matched": 0, "content_restored": 0}
        return cls.enrich_records(records, source)

    @classmethod
    def from_json(cls, path: str | Path) -> list[EventSchema]:
        return cls.from_records(cls.load_records(path))

    to_jsonl = staticmethod(EventAdapter.to_jsonl)
    read_jsonl = staticmethod(EventAdapter.read_jsonl)
