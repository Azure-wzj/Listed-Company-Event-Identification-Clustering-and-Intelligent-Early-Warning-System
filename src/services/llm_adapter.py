"""大模型初评适配层（可插拔，OpenAI 兼容接口）。

职责：把用户输入的自然语言新闻，解析为 结构化事件字段 + 初步分类与初评级。
设计为「可选在线增强」：
  - 配置了 EVENTRADAR_LLM_API_KEY（或 DEEPSEEK_API_KEY）→ 真实调用大模型；
  - 未配置 → parse() 返回 None（降级），由自有模型流水线自行承担解析与纠正，
    保证离线演示不受影响。

支持任意 OpenAI 兼容服务（DeepSeek / Qwen / GLM / OpenAI 等），
通过环境变量配置：
  EVENTRADAR_LLM_BASE_URL  默认 https://api.deepseek.com/v1
  EVENTRADAR_LLM_MODEL     默认 deepseek-chat
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
TIMEOUT = 30.0


def _api_key() -> str | None:
    return os.environ.get("EVENTRADAR_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or None


def _base_url() -> str:
    return os.environ.get("EVENTRADAR_LLM_BASE_URL") or DEFAULT_BASE_URL


def _model() -> str:
    return os.environ.get("EVENTRADAR_LLM_MODEL") or DEFAULT_MODEL


def _resolve(override: dict | None = None) -> tuple[str | None, str, str]:
    """解析 (api_key, base_url, model)：请求级覆盖优先于环境变量/默认值。

    界面传入了密钥时只用于本次调用，不写入服务器磁盘。
    """
    override = override or {}
    key = (override.get("api_key") or "").strip() or _api_key()
    base = (override.get("base_url") or "").strip() or _base_url()
    model = (override.get("model") or "").strip() or _model()
    return key, base, model


def config_status(override: dict | None = None) -> dict:
    key, base, model = _resolve(override)
    return {
        "enabled": bool(key),
        "key": "***" + key[-4:] if key else None,
        "base_url": base,
        "model": model,
    }


def _taxonomy_prompt() -> str:
    """从事件体系文件抽取 64 类事件名 + 17 行业清单，喂给 LLM 约束输出。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    event_names: list[str] = []
    industries: list[str] = []

    comp_path = root / "事件类型_标的.json"
    ind_path = root / "事件类型_行业.json"
    try:
        if comp_path.exists():
            data = json.loads(comp_path.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else []:
                for schema in item.get("event_schema", []):
                    name = schema.get("event_name")
                    if name and name not in event_names:
                        event_names.append(name)
        if ind_path.exists():
            data = json.loads(ind_path.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else []:
                industry = item.get("industry") or {}
                if industry.get("induname"):
                    industries.append(industry["induname"])
                for schema in item.get("event_schema", []):
                    name = schema.get("event_name")
                    if name and name not in event_names:
                        event_names.append(name)
    except Exception:
        pass

    lines = [
        "事件类型必须严格从我方事件体系中选择：",
        json.dumps(event_names, ensure_ascii=False),
        "",
        "行业名称（如事件无公司主体时使用）：",
        json.dumps(industries, ensure_ascii=False),
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """你是金融事件分析助手。把用户提供的新闻内容解析为结构化金融事件，供下游事件雷达系统使用。

只输出一个 JSON 对象，字段如下：
{
  "article_title": "精炼后的新闻标题（20-60字）",
  "content": "新闻要点整理（100-300字，保留时间/主体/数据等关键信息）",
  "article_source": "来源或'用户提交'",
  "trading_code": "如果新闻明确涉及某家上市公司，给出其 6 位证券代码；否则 null",
  "secu_abbr": "对应公司简称，如'比亚迪'；无则 null",
  "industry_name": "如果新闻是行业/政策类（无具体公司），给出行业名称；否则 null",
  "event_name": "事件类型（见事件体系清单）",
  "event_entity": "事件主体一句话",
  "event_impact_direction": "正面/负面/中性",
  "event_time": "事件发生时间 ISO 格式，无法确定则 null",
  "initial_level": "初步预警级别 P0/P1/P2/P3（P0最紧急；单条未证实消息最多给 P2，未涉及上市公司仅为 P3）",
  "rationale": "给出初评级理由的一句话"
}
要求：
- 事件类型严格限定在事件体系清单内，不要自造类型；找不到精确匹配时选最接近的。
- 证券代码必须真实有效；不确定时输出 null，绝不编造。
- initial_level 是「初步印象」，后续由专业模型纠正，尽量保守。"""


def parse_news(text: str, article_source: str = "", override: dict | None = None) -> Optional[dict]:
    """解析自然语言新闻 → 事件字段 dict；未配置 Key 或调用失败返回 None。

    override 可传入界面填写的 {api_key, base_url, model}，仅作用于本次调用。
    """
    key, base_url, model = _resolve(override)
    if not key:
        return None

    taxonomy = _taxonomy_prompt()
    user_prompt = f"{taxonomy}\n\n新闻内容：\n{text}"
    if article_source:
        user_prompt += f"\n（来源：{article_source}）"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as exc:
        # 返回结构化的失败信息，让上层决定（不抛异常）
        return {"_llm_error": f"{type(exc).__name__}: {exc}"}


def test_connection(override: dict | None = None) -> dict:
    """连通性测试：发起一次最小调用，返回是否可用与延迟。界面「测试连接」按钮使用。"""
    key, base_url, model = _resolve(override)
    if not key:
        return {"ok": False, "error": "未填写 API Key"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "回复两个字：正常"}],
        "max_tokens": 8,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {key}"}
    started = time.time()
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        reply = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {
            "ok": True,
            "model": model,
            "base_url": base_url,
            "latency_ms": int((time.time() - started) * 1000),
            "reply": str(reply)[:40],
        }
    except httpx.HTTPStatusError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                "latency_ms": int((time.time() - started) * 1000)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": int((time.time() - started) * 1000)}
