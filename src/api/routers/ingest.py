"""增量事件接入接口（方案 A：轻量归组 + 实时预警，无 torch/BGE）。

POST /api/v1/ingest        接收一条或一批 结构化事件，实时归组并更新预警
GET  /api/v1/ingest/status 增量服务与索引状态
GET  /api/v1/ingest/recent 最近接入的日志（data/web/ingest_log.jsonl）
POST /api/v1/ingest/reset  重置增量状态，回到全量批处理基线（演示用）
"""
import json

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from src.services.ingest_service import get_ingest_service

router = APIRouter()


class IngestResponse(BaseModel):
    total: int
    results: list[dict]


@router.post("")
async def ingest_events(request: Request):
    """接收事件（单条对象 / 列表 / {events:[...]}），返回逐条归组与预警结果。"""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict) and isinstance(payload.get("events"), list):
        events = payload["events"]
    else:
        events = [payload]

    if not events:
        raise HTTPException(status_code=400, detail="请求体为空：请传入事件对象或 {events: [...]}}")

    service = get_ingest_service()
    try:
        results = [service.process_event(item) for item in events[:100]]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"total": len(results), "results": results}


@router.get("/status")
def ingest_status():
    return get_ingest_service().status()


@router.get("/recent")
def ingest_recent(limit: int = Query(20, ge=1, le=500)):
    from src.services.ingest_service import LOG_PATH

    records = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)
    parsed = []
    for line in records[-limit:][::-1]:
        try:
            parsed.append(json.loads(line))
        except ValueError:
            continue
    return {"total": len(records), "records": parsed}


@router.post("/nl")
async def ingest_nl(request: Request):
    """两阶段智能接入：自然语言 → 大模型初评 → 自有模型纠正（BGE归组+B分类+终评）。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="缺少 text 字段（自然语言新闻描述）")
    source = str(body.get("article_source") or "")
    llm_override = body.get("llm") if isinstance(body.get("llm"), dict) else None

    from src.services.nl_pipeline import get_nl_pipeline

    try:
        result = get_nl_pipeline().process(text, source, llm_override)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"完整路径依赖未安装（{exc}）；轻量演示包请使用结构化 JSON 接入",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"两阶段接入失败: {type(exc).__name__}: {exc}")
    return result


@router.post("/nl/test")
async def ingest_nl_test(request: Request):
    """测试大模型连通性：界面「测试连接」按钮使用，不落盘、不保存密钥。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    from src.services import llm_adapter

    override = body.get("llm") if isinstance(body.get("llm"), dict) else body
    return llm_adapter.test_connection(override if isinstance(override, dict) else None)


@router.get("/nl/status")
def ingest_nl_status():
    """两阶段接入状态：LLM 配置、存量组索引、阈值。"""
    try:
        from src.services.nl_pipeline import get_nl_pipeline

        return get_nl_pipeline().status()
    except ImportError as exc:
        return {"available": False, "hint": f"完整路径依赖未安装（{exc}）", "index": {}, "llm": {}}


@router.post("/reset")
def ingest_reset():
    """清空增量接入状态（删日志、重建索引），回到全量批处理基线。"""
    service = get_ingest_service()
    service.reset()
    return service.status()
