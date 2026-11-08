"""
时间线接口 — 从 alerts.json（带评分的事件组）构建真实事件时间线
支持按标的代码 / 事件类型筛选
"""
from fastapi import APIRouter, Query

from src.api.data_store import load_results

router = APIRouter()


def _load_scored_groups():
    return load_results("alerts")


def _to_timeline_item(g: dict) -> dict:
    """事件组 → 时间线条目"""
    return {
        "group_id": g.get("group_id"),
        "time": g.get("timeline_start"),
        "title": g.get("merged_title", ""),
        "summary": (g.get("merged_content", "") or "")[:80],
        "event_name": g.get("event_name"),
        "article_count": g.get("article_count", 0),
        "trading_codes": g.get("trading_codes", []),
        "secu_names": g.get("secu_names", []),
        "alert_level": g.get("alert_level"),
        "credibility_score": g.get("credibility_score"),
        "impact_score": g.get("impact_score"),
        "timeline_stages": g.get("timeline_stages", []),
        "representative_article_id": g.get("representative_article_id"),
    }


@router.get("/")
def get_timeline(
    company: str | None = Query(None, description="标的代码筛选，如 601127"),
    event_type: str | None = Query(None, description="事件类型筛选"),
    limit: int = Query(200, ge=1, le=1000, description="最大返回条数"),
):
    """获取事件时间线（按时间升序）"""
    groups = _load_scored_groups()
    if company:
        groups = [g for g in groups if company in (g.get("trading_codes") or [])]
    if event_type:
        groups = [g for g in groups if g.get("event_name") == event_type]

    timeline = [_to_timeline_item(g) for g in groups]
    timeline.sort(key=lambda x: (x["time"] is not None, x["time"] or ""))
    return {
        "total": len(timeline),
        "timeline": timeline[:limit],
    }


@router.get("/company/{trading_code}")
def get_company_timeline(trading_code: str, limit: int = Query(200, ge=1, le=1000)):
    """获取某公司的完整事件时间线"""
    groups = [g for g in _load_scored_groups() if trading_code in (g.get("trading_codes") or [])]
    timeline = [_to_timeline_item(g) for g in groups]
    timeline.sort(key=lambda x: (x["time"] is not None, x["time"] or ""))
    return {
        "trading_code": trading_code,
        "total": len(timeline),
        "timeline": timeline[:limit],
    }


@router.get("/stats")
def timeline_stats():
    """事件类型 / 公司分布统计。

    未分类事件（无主体且无法确定事件体系范围）不计入类型分布，单独计数，
    以免在图表中出现无名柱状/扇区；其数量通过 unclassified 返回。
    """
    from collections import Counter
    groups = _load_scored_groups()
    event_dist: Counter = Counter()
    company_dist: Counter = Counter()
    unclassified = 0
    for g in groups:
        name = (g.get("event_name") or "").strip()
        if name:
            event_dist[name] += 1
        else:
            unclassified += 1
        for code in g.get("trading_codes", []):
            company_dist[code] += 1
    return {
        "total_groups": len(groups),
        "unclassified": unclassified,
        "by_event_type": dict(event_dist.most_common()),
        "by_company": dict(company_dist.most_common()),
    }
