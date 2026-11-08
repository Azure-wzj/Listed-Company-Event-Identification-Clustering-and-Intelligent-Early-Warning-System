"""
关系图谱接口 — 从 alerts.json 构建「公司—事件」关系图谱。

设计要点：
  - 默认只返回「公司 + 事件」两类节点，去掉来源节点（噪声主要来源）；
  - 所有边按 (source, target, relation) 去重，避免同一事件多篇报道产生重复边；
  - 支持按公司、事件类型、预警等级、最小报道数筛选，控制图规模；
  - 提供公司列表接口，供前端做聚焦式浏览。
"""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query

from src.api.data_store import load_results

router = APIRouter()

ALERT_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _code_name_map() -> dict[str, str]:
    """权威的证券代码 → 简称映射（来自事件体系文件）。"""
    path = _PROJECT_ROOT / "事件类型_标的.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mapping: dict[str, str] = {}
    for item in data if isinstance(data, list) else []:
        company = item.get("company") or {}
        code = str(company.get("trading_code") or "").strip()
        name = str(company.get("secu_abbr") or "").strip()
        if code and name:
            mapping[code] = name
    return mapping


def _load_scored_groups():
    return load_results("alerts")


def _company_pairs(group: dict) -> list[tuple[str, str]]:
    codes = group.get("trading_codes") or []
    names = group.get("secu_names") or []
    authoritative = _code_name_map()
    pairs = []
    for idx, code in enumerate(codes):
        if not code:
            continue
        code = str(code)
        raw_name = names[idx] if idx < len(names) and names[idx] else ""
        name = authoritative.get(code) or str(raw_name) or code
        pairs.append((code, name))
    return pairs


def _group_industry(group: dict) -> str | None:
    """取事件组的行业名（从组内事件字段回退）。"""
    for event in group.get("events", []):
        name = event.get("industry_name")
        if name:
            return str(name)
    return None


def _build_graph(
    groups: list[dict],
    include_sources: bool = False,
    include_event_types: bool = False,
    include_industries: bool = True,
    hide_isolated: bool = True,
) -> dict:
    """构建公司—事件关系图（可选：行业、事件类型、来源），边自动去重。

    行业节点仅在事件没有公司归属时使用，作为行业/政策类事件的归属，
    既避免孤立点，又不给已有公司关联的事件增加噪声边。
    """
    nodes: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}

    for g in groups:
        event_id = g["group_id"]
        nodes[event_id] = {
            "id": event_id,
            "label": (g.get("merged_title", "") or "")[:26],
            "type": "event",
            "event_name": g.get("event_name"),
            "article_count": g.get("article_count", 0),
            "alert_level": g.get("alert_level"),
            "impact_score": g.get("impact_score"),
            "credibility_score": g.get("credibility_score"),
            "cluster_confidence": g.get("cluster_confidence"),
            "representative_article_id": g.get("representative_article_id"),
        }

        company_pairs = _company_pairs(g)
        for code, name in company_pairs:
            nodes.setdefault(code, {
                "id": code,
                "label": name,
                "type": "company",
                "trading_code": code,
            })
            key = (event_id, code, "event-company")
            edges[key] = {
                "source": event_id,
                "target": code,
                "relation": "event-company",
                "event_name": g.get("event_name"),
                "alert_level": g.get("alert_level"),
                "weight": g.get("impact_score") or g.get("article_count", 1),
            }

        if not company_pairs and include_industries:
            industry = _group_industry(g)
            if industry:
                industry_node = f"industry_{industry}"
                nodes.setdefault(industry_node, {
                    "id": industry_node,
                    "label": industry,
                    "type": "industry",
                })
                key = (event_id, industry_node, "belongs_to")
                edges[key] = {
                    "source": event_id,
                    "target": industry_node,
                    "relation": "belongs_to",
                    "event_name": g.get("event_name"),
                    "alert_level": g.get("alert_level"),
                    "weight": 1,
                }

        if include_event_types:
            event_type = g.get("event_name") or "未知"
            type_node = f"type_{event_type}"
            nodes.setdefault(type_node, {
                "id": type_node,
                "label": event_type,
                "type": "event_type",
            })
            key = (event_id, type_node, "has_type")
            edges[key] = {
                "source": event_id,
                "target": type_node,
                "relation": "has_type",
                "weight": 1,
            }

        if include_sources:
            seen_sources = {
                event.get("article_source")
                for event in g.get("events", [])
                if event.get("article_source")
            }
            for source in seen_sources:
                source_node = f"source_{source}"
                nodes.setdefault(source_node, {
                    "id": source_node,
                    "label": source,
                    "type": "source",
                })
                key = (event_id, source_node, "reported_by")
                edges[key] = {
                    "source": event_id,
                    "target": source_node,
                    "relation": "reported_by",
                    "weight": 1,
                }

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _drop_isolated(graph: dict) -> dict:
    """移除没有任何连边的节点。"""
    connected: set[str] = set()
    for edge in graph["edges"]:
        connected.add(edge["source"])
        connected.add(edge["target"])
    return {
        "nodes": [n for n in graph["nodes"] if n["id"] in connected],
        "edges": graph["edges"],
    }


def _filter_groups(
    groups: list[dict],
    company: str | None,
    alert_levels: str | None,
    event_type: str | None,
    only_alerted: bool,
    min_article_count: int,
) -> list[dict]:
    result = groups
    if company:
        result = [g for g in result if company in (g.get("trading_codes") or [])]
    if event_type:
        result = [g for g in result if g.get("event_name") == event_type]
    if alert_levels:
        wanted = {level.strip().upper() for level in alert_levels.split(",") if level.strip()}
        if wanted:
            result = [g for g in result if (g.get("alert_level") or "P3") in wanted]
    elif only_alerted:
        result = [g for g in result if (g.get("alert_level") or "P3") != "P3"]
    if min_article_count > 1:
        result = [g for g in result if (g.get("article_count") or 1) >= min_article_count]
    return result


@router.get("/")
def get_graph(
    company: str | None = Query(None, description="按证券代码聚焦某公司"),
    event_type: str | None = Query(None, description="按事件类型筛选"),
    alert_levels: str | None = Query(None, description="预警等级，逗号分隔，如 P0,P1"),
    only_alerted: bool = Query(False, description="仅保留非 P3 事件"),
    min_article_count: int = Query(1, ge=1, description="最小报道数"),
    include_sources: bool = Query(False, description="是否包含来源节点"),
    include_event_types: bool = Query(False, description="是否包含事件类型节点"),
    include_industries: bool = Query(True, description="无公司归属的事件是否连到行业节点"),
    hide_isolated: bool = Query(True, description="隐藏无任何关联的节点"),
    sort: str = Query("articles", description="排序字段：articles 或 impact"),
    limit: int = Query(60, ge=1, le=300, description="事件组数量上限"),
):
    """获取公司—事件关系图谱（默认紧凑模式）。"""
    groups = _load_scored_groups()
    groups = _filter_groups(
        groups, company, alert_levels, event_type, only_alerted, min_article_count
    )
    total_events = len(groups)

    if sort == "impact":
        groups.sort(key=lambda g: (g.get("impact_score") or 0), reverse=True)
    else:
        groups.sort(key=lambda g: (g.get("article_count") or 0), reverse=True)

    selected = groups[:limit]
    graph = _build_graph(
        selected, include_sources, include_event_types, include_industries
    )
    if hide_isolated:
        graph = _drop_isolated(graph)

    types: dict[str, int] = {}
    for node in graph["nodes"]:
        types[node["type"]] = types.get(node["type"], 0) + 1

    return {
        "total_events": total_events,
        "shown_events": len(selected),
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
        "node_types": types,
        **graph,
    }


@router.get("/companies")
def list_companies(limit: int = Query(50, ge=1, le=200)):
    """公司列表（按关联事件数降序），供前端聚焦选择。"""
    agg: dict[str, dict] = {}
    for g in _load_scored_groups():
        for code, name in _company_pairs(g):
            item = agg.setdefault(code, {
                "code": code,
                "name": name,
                "events": 0,
                "alerted": 0,
                "articles": 0,
            })
            item["events"] += 1
            item["articles"] += g.get("article_count") or 1
            if (g.get("alert_level") or "P3") != "P3":
                item["alerted"] += 1
    companies = sorted(agg.values(), key=lambda x: x["events"], reverse=True)
    return {"total": len(companies), "companies": companies[:limit]}


@router.get("/company/{trading_code}")
def get_company_graph(
    trading_code: str,
    limit: int = Query(80, ge=1, le=300),
    include_sources: bool = Query(False),
):
    """获取某公司的关联事件子图（始终返回该公司的全部事件，按报道数降序取前 limit）。"""
    groups = [g for g in _load_scored_groups() if trading_code in (g.get("trading_codes") or [])]
    groups.sort(key=lambda g: (g.get("article_count") or 0), reverse=True)
    graph = _build_graph(groups[:limit], include_sources=include_sources)
    return {
        "trading_code": trading_code,
        "total_events": len(groups),
        "shown_events": len(groups[:limit]),
        **graph,
    }
