"""两阶段纠错增量接入：LLM 初评 → 自有模型纠正。

链路（与存量数据同一条决策链）：
  1) 用户输入自然语言新闻；
  2) 大模型初评（可插拔，未配 Key 时跳过）：结构化字段 + 初步分类 + 初评级；
  3) 自有模型纠正：
     a. BGE 嵌入 → 存量 已有事件组代表向量索引 → 归组或新建；
     b. 事件分类器纠正事件类型 / 影响方向（映射到事件体系）；
     c. 新建组走原流水线同款 事件评分（可信度/影响力/情绪）；
     d. AlertEngine 终评（覆盖 LLM 初评级，并输出纠偏说明）；
  4) 结果写回 alerts/event_groups 双缓存（各页面立即可见）+ 持久化。

完整模式依赖（可选）：torch + BGE + 存量组索引(scripts/build_group_index.py)。
未满足时返回明确提示并降级，不影响离线演示。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.api.schemas.event import EventSchema
from src.services import llm_adapter
from src.services.alert_engine import AlertEngine
from src.services.ingest_service import _event_snapshot, _event_text, _validated, find_group_in_caches
from src.services.group_vector_index import DEFAULT_THRESHOLD, get_group_index

ROOT = Path(__file__).resolve().parents[2]
NL_LOG = ROOT / "data" / "web" / "nl_log.jsonl"

FULL_MODE_HINT = "完整路径需要 BGE 模型与存量组索引（scripts/build_group_index.py）"


class NlPipeline:
    """单例；线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embedder = None
        self._classifier = None
        self._seq = 0
        self._article_owner: dict[str, str] = {}
        self.stats = {"processed": 0, "matched": 0, "new": 0, "llm_error": 0, "error": 0}
        self.last_results: list[dict] = []

    # ── 懒加载 ─────────────────────────────────────────
    @property
    def embedder(self):
        if self._embedder is None:
            from src.features.embedding import EmbeddingEngine

            self._embedder = EmbeddingEngine()
        return self._embedder

    @staticmethod
    def _event_text(event: EventSchema) -> str:
        return _event_text(event)

    @property
    def classifier(self):
        if self._classifier is None:
            # 惰性导入：轻量演示包未安装 pandas/gensim 时不影响其它接口
            from src.services.event_classifier import get_default_event_classifier

            self._classifier = get_default_event_classifier()
        return self._classifier

    def status(self) -> dict:
        index = get_group_index()
        if not index.ready:
            index.load()  # 懒加载索引
        llm = llm_adapter.config_status()
        return {
            "available": index.ready,
            "hint": None if index.ready else index.error or FULL_MODE_HINT,
            "index": index.status(),
            "llm": llm,
            "threshold": DEFAULT_THRESHOLD,
            "stats": dict(self.stats),
        }

    # ── 文本 → 向量（与原流水线同款 BGE） ─────────────
    def _embed_group_text(self, event: EventSchema) -> np.ndarray:
        return self.embedder.encode([self._event_text(event)], show_progress=False)[0]

    # ── 归组（查存量组索引） ───────────────────────────
    def _match_group(self, vec: np.ndarray) -> tuple[int | None, float]:
        index = get_group_index()
        if not index.ready:
            index.load()
            if not index.ready:
                return None, 0.0
        top = index.topk(vec, k=3)
        if not top:
            return None, 0.0
        pos, sim = top[0]
        if sim >= DEFAULT_THRESHOLD:
            return pos, sim
        return None, sim

    # ── 事件分类 ─────────────────────────────────────
    def _correct(self, event: EventSchema, llm_name: str | None) -> dict:
        """用 事件分类器纠正事件类型与影响方向（映射到事件体系 taxonomy）。"""
        group = _one_event_group(event)
        classification = self.classifier.classify_group(group)
        if classification is None:
            return {"event_name": llm_name, "impact_direction": event.event_impact_direction,
                    "confidence": None, "changed": False, "reason": "分类器未给出结果"}
        corrected = classification.event_name
        direction = classification.impact_direction
        changed = bool(llm_name and corrected and corrected != llm_name)
        reason = (
            f"事件分类器判定为「{corrected}」"
            + (f"，纠正 LLM 初判「{llm_name}」" if changed else "，与 LLM 初判一致")
            + f"（置信度 {classification.confidence}）"
        )
        return {
            "event_name": corrected,
            "impact_direction": direction,
            "confidence": classification.confidence,
            "classifier": classification.classifier,
            "changed": changed,
            "reason": reason,
        }

    # ── 核心处理 ───────────────────────────────────────
    def process(self, text: str, article_source: str = "", llm_override: dict | None = None) -> dict:
        """处理一条自然语言新闻，返回两阶段结果。

        llm_override 为界面填写的 {api_key, base_url, model}，仅作用于本次调用。
        """
        with self._lock:
            return self._process_locked(text, article_source, llm_override)

    def _process_locked(self, text: str, article_source: str, llm_override: dict | None = None) -> dict:
        self._seq += 1
        start = time.time()
        steps: list[dict] = []
        llm_error = None

        # 1. LLM 初评（可插拔）
        llm = llm_adapter.parse_news(text, article_source, llm_override)
        if llm is None:
            llm = {}
        elif llm.get("_llm_error"):
            llm_error = llm.pop("_llm_error")
            self.stats["llm_error"] += 1
        steps.append({"step": "llm", "label": "大模型初评",
                      "detail": llm if llm else ("未配置 API Key，跳过（由自有模型直接解析纠正）" if not llm_error else llm_error)})

        # 2. 组装事件（结构化）
        try:
            event = EventSchema(
                article_file_id=f"NL{int(time.time())}-{self._seq}",
                article_title=(llm.get("article_title") or text[:60]),
                content=(llm.get("content") or text),
                article_publish_time=llm.get("event_time") or datetime.now().isoformat(),
                event_time=llm.get("event_time"),
                article_source=(llm.get("article_source") or article_source or "用户提交"),
                trading_code=llm.get("trading_code"),
                secu_abbr=llm.get("secu_abbr"),
                industry_name=llm.get("industry_name"),
                event_name=llm.get("event_name"),
                event_entity=llm.get("event_entity"),
                event_impact_direction=llm.get("event_impact_direction"),
            )
        except Exception as exc:
            self.stats["error"] += 1
            return self._wrap_result(steps, error=f"事件组装失败: {exc}", elapsed=time.time() - start)

        # 查重
        owner = self._article_owner.get(event.article_file_id)
        if owner:
            result = self._wrap_result(steps, event=event, duplicate_group=owner,
                                       elapsed=time.time() - start)
            self.last_results.insert(0, result)
            return result

        # 3. 自有模型纠正（先算向量；无完整路径时降级到 TF-IDF，而不是直接失败）
        vec = None
        bge_error = None
        index = get_group_index()
        if not index.ready:
            index.load()
        if index.ready:
            try:
                vec = self._embed_group_text(event)
            except Exception as exc:
                bge_error = f"{type(exc).__name__}: {exc}"
        else:
            bge_error = index.error or FULL_MODE_HINT

        # 3a. 归组
        if vec is not None:
            pos, sim = self._match_group(vec)
            matched_group_id = index.group_ids[pos] if pos is not None else None
            if matched_group_id:
                steps.append({"step": "cluster", "label": "BGE 归组",
                              "detail": f"命中存量组 {matched_group_id}（相似度 {sim:.3f}，阈值 {DEFAULT_THRESHOLD}）"})
            else:
                steps.append({"step": "cluster", "label": "BGE 归组",
                              "detail": f"未命中存量组（最高相似度 {sim:.3f} < {DEFAULT_THRESHOLD}），将新建事件组"})
        else:
            # 轻量降级：未启用 BGE（如演示包未安装 torch / 未生成存量组索引）
            matched_group_id = None
            sim = None
            steps.append({"step": "cluster", "label": "归组（轻量）",
                          "detail": "未启用 BGE 完整路径，改用字符 n-gram TF-IDF 归组"})
            steps.append({"step": "hint", "label": "完整路径说明",
                          "detail": f"{FULL_MODE_HINT}；当前由轻量路径完成归组与预警"})

        # 3b. 事件分类
        try:
            correction = self._correct(event, llm.get("event_name"))
            steps.append({"step": "b_classify", "label": "事件分类纠正",
                          "detail": correction["reason"]})
        except Exception as exc:
            correction = {"event_name": event.event_name, "impact_direction": event.event_impact_direction,
                          "confidence": None, "changed": False,
                          "reason": "轻量模式未启用事件分类器（完整依赖未安装），保留原始事件类型"}
            steps.append({"step": "b_classify", "label": "事件分类纠正", "detail": correction["reason"], "warn": True})

        snapshot = _event_snapshot(event)
        snapshot["event_name"] = correction["event_name"]
        snapshot["event_impact_direction"] = correction["impact_direction"]

        if vec is None:
            # 轻量降级：无 BGE 与存量组索引时，改用 TF-IDF 归组 + 预警（演示包同样可用）
            return self._finish_lightweight(event, correction, steps, payload_extra=None,
                                            llm=llm, llm_error=llm_error,
                                            text=text, article_source=article_source,
                                            start=start)

        if matched_group_id:
            # 归入存量组：并入双缓存，预警沿用存量批处理口径
            alert_group, event_group = find_group_in_caches(matched_group_id)
            targets = []
            for g in (alert_group, event_group):
                if g is not None and not any(g is t for t in targets):
                    targets.append(g)
            for g in targets:
                g.setdefault("events", []).append(snapshot)
                g["article_count"] = (g.get("article_count") or 0) + 1
                sources = {e.get("article_source") for e in g["events"] if e.get("article_source")}
                g["source_diversity"] = round(len(sources) / len(g["events"]), 4)
                ts = snapshot.get("article_publish_time")
                if ts and (not g.get("timeline_end") or str(ts) > str(g["timeline_end"])):
                    g["timeline_end"] = ts
                if event.trading_code and event.trading_code not in (g.get("trading_codes") or []):
                    g.setdefault("trading_codes", []).append(event.trading_code)
                    g.setdefault("secu_names", []).append(event.secu_abbr or event.trading_code)
            self._article_owner[event.article_file_id] = matched_group_id
            self.stats["matched"] += 1
            alert = {
                "level": (alert_group or {}).get("alert_level"),
                "note": "归入存量事件组，预警沿用全量批处理评分",
            }
            message = f"已归入存量事件组 {matched_group_id}"
            steps.append({"step": "alert", "label": "预警终评",
                          "detail": f"存量组预警 {alert['level']}（沿用批处理评分，不因单条报道改动）"})
        else:
            # 新建组：原流水线同款 事件评分 + AlertEngine 终评
            new_group_id = f"I{self._seq:06d}"
            group = _one_event_group(event, group_id=new_group_id)
            group = group.model_copy(update={
                "group_id": new_group_id,
                "merged_title": event.article_title or "(无标题)",
                "merged_content": (event.content or "")[:500],
                "article_count": 1,
                "source_diversity": 1.0,
                "event_name": correction["event_name"],
                "trading_codes": [event.trading_code] if event.trading_code else [],
                "secu_names": [event.secu_abbr] if event.secu_abbr else ([event.trading_code] if event.trading_code else []),
                "representative_article_id": event.article_file_id,
                "cluster_confidence": 1.0,
                "timeline_start": snapshot.get("article_publish_time"),
                "timeline_end": snapshot.get("article_publish_time"),
                "cluster_reason": "自然语言接入新建组（未命中存量组）",
            })
            scored = _score_group(group)  # 原流水线 EventScorer（含分类器）
            final = AlertEngine.apply(scored)
            dumped = final.model_dump(mode="json")
            cache_group = {
                k: dumped.get(k) for k in (
                    "group_id", "events", "cluster_confidence", "timeline_start", "timeline_end",
                    "merged_title", "merged_content", "article_count", "source_diversity",
                    "event_name", "trading_codes", "secu_names", "representative_article_id",
                    "evidence_edges", "cluster_reason", "timeline_stages",
                    "credibility_score", "impact_score", "impact_level", "sentiment_score",
                    "b_score_components", "b_score_reason", "b_alert_level_hint", "b_match_method",
                    "b_match_score", "alert_level", "alert_reason", "alert_factors",
                    "previous_alert_level", "alert_changed_reason",
                )
            }
            from src.api.data_store import load_results

            load_results("alerts").append(cache_group)
            load_results("event_groups").append(cache_group)
            self._article_owner[event.article_file_id] = new_group_id
            self.stats["new"] += 1
            alert = {"level": final.alert_level, "reason": final.alert_reason,
                     "changed": final.alert_changed_reason}
            message = f"已新建事件组 {new_group_id}，终评预警 {final.alert_level}"
            steps.append({"step": "alert", "label": "预警终评",
                          "detail": f"影响分 {final.impact_score}，可信度 {final.credibility_score} → 预警 {final.alert_level}（覆盖 LLM 初评）"})

        # 4. 持久化
        self._append_log({"text": text, "article_source": article_source, "seq": self._seq})

        self.stats["processed"] += 1
        result = self._wrap_result(steps, event=event, alert=alert, matched_group_id=matched_group_id,
                                   new_group_id=None if matched_group_id else new_group_id,
                                   message=message, sim=sim if matched_group_id else None,
                                   llm_error=llm_error, elapsed=time.time() - start,
                                   llm_initial=llm)
        self.last_results.insert(0, result)
        self.last_results = self.last_results[:50]
        return result

    def _finish_lightweight(self, event: EventSchema, correction: dict, steps: list[dict],
                            payload_extra, llm: dict, llm_error: str | None,
                            text: str, article_source: str, start: float) -> dict:
        """轻量路径收尾：交由 IngestService（字符 n-gram TF-IDF）完成归组与预警。"""
        from src.services.ingest_service import get_ingest_service

        payload = {
            "article_file_id": event.article_file_id,
            "article_title": event.article_title,
            "article_source": event.article_source,
            "content": event.content,
            "event_name": correction.get("event_name") or event.event_name,
            "event_impact_direction": correction.get("impact_direction") or event.event_impact_direction,
            "trading_code": event.trading_code,
            "secu_abbr": event.secu_abbr,
            "industry_name": event.industry_name,
            "article_publish_time": event.article_publish_time.isoformat() if event.article_publish_time else None,
        }
        res = get_ingest_service().process_event(payload)
        status = res.get("status")
        group_id = res.get("matched_group_id")
        alert = res.get("alert") or {}
        steps.append({"step": "alert", "label": "预警终评",
                      "detail": f"{res.get('message', '')}；预警 {alert.get('level', '-')}"})

        self._article_owner[event.article_file_id] = group_id
        self.stats["matched" if status == "matched" else "new"] += 1
        self.stats["processed"] += 1
        self._append_log({"text": text, "article_source": article_source, "seq": self._seq})

        result = self._wrap_result(
            steps, event=event, alert=alert,
            matched_group_id=group_id if status == "matched" else None,
            new_group_id=group_id if status == "new" else None,
            duplicate_group=group_id if status == "duplicate" else None,
            message=f"轻量路径（TF-IDF）完成：{res.get('message', '')}",
            sim=res.get("similarity"), llm_initial=llm, llm_error=llm_error,
            elapsed=time.time() - start,
        )
        self.last_results.insert(0, result)
        self.last_results = self.last_results[:50]
        return result

    def _append_log(self, record: dict) -> None:
        try:
            NL_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(NL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _wrap_result(self, steps: list[dict], event: EventSchema | None = None,
                     alert: dict | None = None, matced_group_id: str | None = None,
                     new_group_id: str | None = None, matched_group_id: str | None = None,
                     message: str = "", sim: float | None = None,
                     llm_initial: dict | None = None, llm_error: str | None = None,
                     duplicate_group: str | None = None, error: str | None = None,
                     hint: str | None = None, elapsed: float = 0.0) -> dict:
        return {
            "ok": not error,
            "error": error,
            "hint": hint,
            "elapsed_ms": round(elapsed * 1000),
            "article": {
                "article_file_id": event.article_file_id if event else None,
                "article_title": event.article_title if event else None,
                "trading_code": event.trading_code if event else None,
                "secu_abbr": event.secu_abbr if event else None,
                "industry_name": event.industry_name if event else None,
                "event_name": event.event_name if event else None,
                "event_impact_direction": event.event_impact_direction if event else None,
            },
            "llm": {
                "initial": llm_initial or {},
                "error": llm_error,
            },
            "steps": steps,
            "status": "duplicate" if duplicate_group else ("matched" if matched_group_id else ("new" if new_group_id else ("error" if error else "ok"))),
            "matched_group_id": matched_group_id,
            "new_group_id": new_group_id,
            "duplicate_group": duplicate_group,
            "similarity": round(sim, 4) if sim is not None else None,
            "alert": alert or {},
            "message": message,
        }


def _one_event_group(event: EventSchema, group_id: str = "tmp") -> Any:
    """构造单事件组（供 事件分类与 事件评分）。"""
    from src.api.schemas.cluster import EventGroupSchema

    return EventGroupSchema(
        group_id=group_id,
        events=[event],
        merged_title=event.article_title or "",
        merged_content=(event.content or "")[:500],
        article_count=1,
        source_diversity=1.0,
        event_name=event.event_name,
        trading_codes=[event.trading_code] if event.trading_code else [],
        secu_names=[event.secu_abbr] if event.secu_abbr else ([event.trading_code] if event.trading_code else []),
        representative_article_id=event.article_file_id,
        cluster_confidence=1.0,
    )


def _score_group(group: Any) -> Any:
    """原流水线同款评分：EventScorer（含 事件分类后的 taxonomy 匹配）→ ScoredEventGroupSchema。"""
    from src.services.event_scorer import EventScorer
    from src.services.event_classifier import get_default_event_classifier

    return EventScorer.score(group, classifier=get_default_event_classifier())


_singleton = None
_singleton_lock = threading.Lock()


def get_nl_pipeline() -> NlPipeline:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = NlPipeline()
        return _singleton
