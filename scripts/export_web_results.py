"""导出 API/前端专用的轻量结果文件。

全量结果（data/results/*.json）包含每篇文章的完整正文，单文件数百 MB，
API 每次请求都需完整解析，导致 10s+ 响应。前端只展示标题、时间、事件类型、
评分与预警等字段，不需要正文。本脚本剥离大字段后 gzip 写入 data/web/
（体积约为原文件 10%），供 API 快速加载（配合进程内缓存）。

用法：
    python -X utf8 scripts/export_web_results.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"
WEB_DIR = ROOT / "data" / "web"

# 每个文件读取来源（scored_groups 无路由使用，不导出）
SOURCES = {
    "event_groups": "event_groups.json",
    "alerts": "alerts.json",
}

# 需要从每个事件（event）中剥离的大字段
EVENT_DROP_FIELDS = ("content",)
# 组级需要保留但截断的字段
TRUNCATE_FIELDS = {"merged_content": 240}


def _slim_event(event: dict) -> dict:
    if not isinstance(event, dict):
        return event
    return {k: v for k, v in event.items() if k not in EVENT_DROP_FIELDS}


def _slim_group(group: dict) -> dict:
    if not isinstance(group, dict):
        return group
    slim = dict(group)
    events = slim.get("events")
    if isinstance(events, list):
        slim["events"] = [_slim_event(e) for e in events]
    for field, limit in TRUNCATE_FIELDS.items():
        value = slim.get(field)
        if isinstance(value, str) and len(value) > limit:
            slim[field] = value[:limit]
    return slim


def export(name: str, filename: str) -> tuple[int, int]:
    src = RESULTS_DIR / filename
    if not src.exists():
        print(f"[skip] {src} 不存在")
        return 0, 0
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    slim = [_slim_group(g) for g in data] if isinstance(data, list) else data

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEB_DIR / f"{filename}.gz"
    payload = json.dumps(slim, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(dst, "wb", compresslevel=6) as f:
        f.write(payload)

    before = src.stat().st_size
    after = dst.stat().st_size
    count = len(slim) if isinstance(slim, list) else 1
    print(
        f"[ok] {filename}: {count:,} 条 | "
        f"{before/1024/1024:.1f}MB -> {after/1024/1024:.1f}MB (gzip)"
    )
    return before, after


def main() -> None:
    total_before = total_after = 0
    for name, filename in SOURCES.items():
        b, a = export(name, filename)
        total_before += b
        total_after += a
    print(
        f"\n合计: {total_before/1024/1024:.1f}MB -> {total_after/1024/1024:.1f}MB "
        f"({total_after/max(total_before,1)*100:.1f}%)"
    )
    print(f"输出目录: {WEB_DIR}")


if __name__ == "__main__":
    main()
