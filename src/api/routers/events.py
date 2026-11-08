"""
事件组查询接口 — 读取 事件聚类产物（event_groups.json）返回真实数据
"""
from fastapi import APIRouter, Query

from src.api.data_store import load_results

router = APIRouter()


def _load_groups():
    return load_results("event_groups")


@router.get("/")
def list_events(
    limit: int = Query(50, ge=1, le=500, description="返回条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """获取事件组列表（真实聚类结果）"""
    groups = _load_groups()
    return {
        "total": len(groups),
        "events": groups[offset: offset + limit],
    }


@router.get("/{group_id}")
def get_event(group_id: str):
    """获取单个事件组详情"""
    for g in _load_groups():
        if g.get("group_id") == group_id:
            return g
    return {"error": f"事件组 {group_id} 不存在"}
