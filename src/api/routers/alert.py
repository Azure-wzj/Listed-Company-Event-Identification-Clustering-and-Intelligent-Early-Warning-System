"""
预警接口 — 读取 预警产物（alerts.json），可按级别过滤
"""
from fastapi import APIRouter, Query
from typing import Literal

from src.api.data_store import load_results

router = APIRouter()


def _load_alerts():
    return load_results("alerts")


@router.get("/")
def list_alerts(
    level: str | None = Query(None, description="按预警级别过滤: P0/P1/P2/P3"),
    sort_by: Literal["impact"] | None = Query(None, description="按影响评分从高到低排序"),
    limit: int = 50,
    offset: int = 0,
):
    """获取预警列表（真实结果，可按级别过滤）"""
    limit = max(1, min(500, int(limit)))
    offset = max(0, int(offset))
    alerts = _load_alerts()
    if level:
        alerts = [a for a in alerts if a.get("alert_level") == level]
    if sort_by == "impact":
        alerts = sorted(alerts, key=lambda a: a.get("impact_score") or 0, reverse=True)
    return {
        "total": len(alerts),
        "alerts": alerts[offset:offset + limit],
    }


@router.get("/stats")
def alert_stats():
    """预警级别分布统计"""
    from collections import Counter
    dist = Counter((a.get("alert_level") or "P3") for a in _load_alerts())
    distribution = {
        "P0": dist.get("P0", 0),
        "P1": dist.get("P1", 0),
        "P2": dist.get("P2", 0),
        "P3": dist.get("P3", 0),
    }
    for key, value in dist.items():
        if key not in distribution:
            distribution[key] = value
    return {"total": sum(distribution.values()), "distribution": distribution}


@router.get("/config")
def get_alert_config():
    """Return active alert thresholds."""
    from src.config import settings
    return {
        "P0": {
            "min_impact": settings.alert_p0_min_impact,
            "min_credibility": settings.alert_p0_min_credibility,
        },
        "P1": {
            "min_impact": settings.alert_p1_min_impact,
            "min_credibility": settings.alert_p1_min_credibility,
        },
        "P2": {
            "min_impact": settings.alert_p2_min_impact,
            "min_credibility": settings.alert_p2_min_credibility,
        },
        "P3": {
            "min_impact": settings.alert_p3_min_impact,
            "min_credibility": settings.alert_p3_min_credibility,
        },
    }
