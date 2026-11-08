"""结果文件加载层：优先读取轻量版（data/web/），带 mtime 感知的进程内缓存。

全量结果文件（data/results/）含正文，数百 MB；轻量版由
scripts/export_web_results.py 生成（.json.gz，约为原体积 10%）。
API 只读轻量版即可满足前端需求，首次解析后缓存，后续请求接近零延迟。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
WEB_DIR = ROOT / "data" / "web"

_cache: dict[str, tuple[float, list]] = {}
_stats: dict[str, dict] = {}


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _resolve(name: str) -> Path:
    """查找结果文件：轻量 .json -> 轻量 .json.gz -> 全量 .json。"""
    for candidate in (
        WEB_DIR / f"{name}.json",
        WEB_DIR / f"{name}.json.gz",
        RESULTS_DIR / f"{name}.json",
    ):
        if candidate.exists():
            return candidate
    return RESULTS_DIR / f"{name}.json"


# 同源视图：某些结果文件是另一文件的子集（字段完全一致，仅少若干列）。
# 避免重复存储，缺失时由源文件派生视图（共享同一批 dict，避免重复内存）。
_ALIAS_VIEWS = {"event_groups": "alerts"}


def load_results(name: str) -> list:
    """加载结果文件（轻量版优先），按 mtime 缓存。

    当 `name` 属于同源视图且自身文件缺失时，返回源数据列表的副本
    （元素为同一 dict 对象，既保持字段一致，又不会因列表同一而重复写入）。
    """
    source = _ALIAS_VIEWS.get(name)
    # A web-only alerts file is a complete scored-group view. Prefer its
    # matching derived view over a potentially stale full result file.
    web_source_exists = source is not None and any(
        (WEB_DIR / f"{source}{suffix}").exists() for suffix in (".json", ".json.gz")
    )
    web_view_exists = any((WEB_DIR / f"{name}{suffix}").exists() for suffix in (".json", ".json.gz"))
    derive_from_web = web_source_exists and not web_view_exists
    if source is not None and (derive_from_web or not _resolve(name).exists()):
        base = load_results(source)
        view = list(base)
        _cache[name] = (_cache.get(source, (0.0, []))[0], view)
        _stats[name] = {
            "path": f"(view of {source})",
            "size_mb": _stats.get(source, {}).get("size_mb", 0.0),
            "records": len(view),
        }
        return view

    path = _resolve(name)
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    cached = _cache.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    with _open_text(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("events", []) if isinstance(data, dict) else []
    _cache[name] = (mtime, data)
    _stats[name] = {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "records": len(data),
    }
    return data


def cache_info() -> dict:
    """已加载文件的来源与规模，便于健康检查/排障。"""
    return {
        "loaded": _stats,
        "using_web_dir": WEB_DIR.exists(),
    }


def clear_cache() -> None:
    _cache.clear()
    _stats.clear()
