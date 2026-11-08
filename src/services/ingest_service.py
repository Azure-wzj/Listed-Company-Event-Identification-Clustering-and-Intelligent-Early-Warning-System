"""轻量增量事件接入服务（方案 A：无 torch / BGE 依赖）。

设计：
  - 归组用字符 n-gram TF-IDF（自带实现，见 char_tfidf.py）代替 BGE 实时向量化：
    启动时对全部事件组的代表文本建稀疏索引（内存 ~几十 MB，秒级构建）；
  - 新事件进来即与索引做余弦匹配：
      * 相似度 ≥ 阈值 → 归入已有组（更新计数 / 来源 / 时间线，重算预警）；
      * 低于阈值 → 创建新组（I 前缀），用「报道佐证」启发式动态评分；
  - 归组结果直接写回 data_store 缓存，前端所有页面立即可见；
  - 每条接入记录持久化到 data/web/ingest_log.jsonl，重启后重放恢复状态。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.api.data_store import _resolve, clear_cache, load_results
from src.api.schemas.event import EventSchema
from src.services.alert_engine import AlertEngine

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "data" / "web" / "ingest_log.jsonl"
# 预构建的轻量归组索引缓存（由 scripts/build_ingest_index.py 生成）
INDEX_CACHE = ROOT / "data" / "cache" / "ingest_index.npz"

# TF-IDF 字符相似度的归组阈值（与 BGE 的 0.94 不是同一量纲，独立标定）
DEFAULT_THRESHOLD = 0.42
MAX_RECENT = 200
TOP_K = 5


def _group_text(group: dict) -> str:
    """事件组代表文本：标题加权（×2）+ 事件类型 + 公司 + 内容摘要。"""
    title = group.get("merged_title") or ""
    content = (group.get("merged_content") or "")[:200]
    secu = " ".join(group.get("secu_names") or [])
    return " ".join(p for p in (title, title, group.get("event_name") or "", secu, content) if p)


def _event_text(event: EventSchema) -> str:
    """新事件查询文本：与组文本同构，标题加权。"""
    title = event.article_title or ""
    content = (event.content or "")[:200]
    return " ".join(
        p for p in (
            title, title, event.event_name or "", event.secu_abbr or "",
            event.industry_name or "", content,
        ) if p
    )


def _event_snapshot(event: EventSchema) -> dict:
    """入库用的事件字段（与 web 版嵌套事件结构一致）。"""
    return {
        "article_file_id": event.article_file_id,
        "article_title": event.article_title,
        "article_publish_time": event.article_publish_time.isoformat() if event.article_publish_time else None,
        "article_source": event.article_source,
        "trading_code": event.trading_code,
        "secu_abbr": event.secu_abbr,
        "industry_code": event.industry_code,
        "industry_name": event.industry_name,
        "event_name": event.event_name,
        "event_time": event.event_time.isoformat() if event.event_time else None,
        "event_entity": event.event_entity,
        "event_impact_direction": event.event_impact_direction,
    }


def _impact_level_pair(article_count: int, source_diversity: float) -> tuple[float, str]:
    """增量组的影响力启发式：报道佐证越多、来源越多样，影响越大。

    返回 (impact_score, impact_level)。单篇报道 → P3；多篇多源佐证后逐步升至 P2/P1/P0。
    """
    impact = round(min(0.95, 0.32 + 0.14 * (article_count - 1) + 0.10 * source_diversity), 4)
    level = "low" if impact < 0.52 else ("moderate" if impact < 0.80 else "high")
    return impact, level


class IngestService:
    """进程内单例；线程安全（FastAPI 线程池下并发请求）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ready = False
        self._building = False
        self._index_error: str | None = None
        self._vectorizer = None
        self._matrix = None
        self._group_ids: list[str] = []
        self._article_owner: dict[str, str] = {}
        self._new_group_seq = 0
        self.threshold = DEFAULT_THRESHOLD
        self.stats = {"matched": 0, "new": 0, "duplicate": 0, "error": 0}

    # ── 索引 ──────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self._ready

    def status(self) -> dict:
        # 无锁读：字段均为原子替换，避免构建期间阻塞状态查询
        return {
            "ready": self._ready,
            "building": self._building,
            "error": self._index_error,
            "groups_indexed": len(self._group_ids),
            "articles_indexed": len(self._article_owner),
            "vocab_size": len(self._vectorizer.vocabulary_) if self._vectorizer else 0,
            "match_threshold": self.threshold,
            "new_group_seq": self._new_group_seq,
            "stats": dict(self.stats),
        }

    def build_index(self) -> None:
        """准备字符 n-gram TF-IDF 组索引。

        构建在**独立子进程**中完成（scripts/build_ingest_index.py），结果缓存为
        data/cache/ingest_index.npz：
          - 避免长时间高分配量的 Python 循环在个别环境下崩溃影响 API 主进程；
          - 缓存命中时启动只需秒级。
        若构建失败，服务仍可用，但降级为“不匹配存量组、直接新建事件组”。
        """
        with self._lock:
            if self._ready or self._building:
                return
            self._building = True
            try:
                groups = load_results("alerts")
                self._group_ids = [g.get("group_id") for g in groups]

                article_owner: dict[str, str] = {}
                new_group_seq = 0
                for g in groups:
                    gid = g.get("group_id") or ""
                    if gid.startswith("I") and gid[1:].isdigit():
                        try:
                            new_group_seq = max(new_group_seq, int(gid[1:]))
                        except ValueError:
                            pass
                    for ev in g.get("events", []):
                        aid = ev.get("article_file_id")
                        if aid:
                            article_owner[str(aid)] = gid
                self._article_owner = article_owner
                self._new_group_seq = new_group_seq

                loaded = self._load_prebuilt()
                if loaded is None and self._build_via_subprocess():
                    loaded = self._load_prebuilt()

                if loaded is None:
                    self._vectorizer = None
                    self._matrix = None
                    self._index_error = (
                        "轻量归组索引不可用（子进程构建失败）：新事件将直接新建事件组"
                    )
                else:
                    vectorizer, matrix, group_ids = loaded
                    self._vectorizer = vectorizer
                    self._matrix = matrix
                    if group_ids:
                        self._group_ids = group_ids
                    self._index_error = None
                self._ready = True
            except Exception as exc:  # 索引失败不应拖垮 API
                import traceback

                self._index_error = (
                    f"{type(exc).__name__}: {exc}\n"
                    + "\n".join(traceback.format_exc().splitlines()[-6:])
                )
            finally:
                self._building = False

    # ── 索引构建：缓存优先 + 子进程隔离 ─────────────────────
    def _load_prebuilt(self):
        """加载缓存索引（需比结果文件新），返回 (vectorizer, matrix, group_ids) 或 None。"""
        import numpy as np
        import scipy.sparse as sp

        from src.services.char_tfidf import CharNgramTfidf

        path = INDEX_CACHE
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            source = _resolve("alerts")
            cached_mtime = float(data["source_mtime"][0]) if "source_mtime" in data else 0.0
            if source.exists() and cached_mtime and abs(source.stat().st_mtime - cached_mtime) > 1.0:
                return None  # 结果数据已更新，缓存过期

            vocab = [str(g) for g in data["vocab"]]
            vectorizer = CharNgramTfidf()
            vectorizer.vocabulary_ = {gram: idx for idx, gram in enumerate(vocab)}
            vectorizer.idf_ = data["idf"].astype(np.float32)
            shape = tuple(int(x) for x in data["shape"])
            matrix = sp.csr_matrix(
                (data["csr_data"], data["csr_indices"], data["csr_indptr"]), shape=shape
            )
            group_ids = [str(g) for g in data["group_ids"]]
            if len(group_ids) != shape[0]:
                group_ids = []
            return vectorizer, matrix, group_ids
        except Exception:
            return None

    @staticmethod
    def _build_via_subprocess() -> bool:
        """在子进程中构建索引（最多重试一次），返回是否成功。"""
        import subprocess
        import sys

        script = ROOT / "scripts" / "build_ingest_index.py"
        if not script.exists():
            print("[ingest] 缺少 scripts/build_ingest_index.py，无法构建索引")
            return False
        for attempt in (1, 2):
            try:
                proc = subprocess.run(
                    [sys.executable, "-X", "utf8", str(script), str(INDEX_CACHE)],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                if proc.returncode == 0 and INDEX_CACHE.exists():
                    tail = (proc.stdout or "").strip().splitlines()[-1:]
                    print(f"[ingest] 索引子进程完成（尝试 {attempt}）{tail}")
                    return True
                err = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
                print(f"[ingest] 索引子进程未成功（尝试 {attempt}, 返回码 {proc.returncode}）{err}")
            except Exception as exc:
                print(f"[ingest] 索引子进程异常（尝试 {attempt}）: {type(exc).__name__}: {exc}")
        return False

    # ── 工具 ──────────────────────────────────────────────
    def _ensure_ready(self) -> None:
        if not self._ready:
            self.build_index()
            if not self._ready:
                raise RuntimeError(f"增量索引不可用: {self._index_error}")

    def _find_group(self, group_id: str) -> tuple[dict | None, dict | None]:
        """在 alerts 与 event_groups 两份缓存中查找组 dict（同一对象引用）。"""
        alert_hit = event_hit = None
        for g in load_results("alerts"):
            if g.get("group_id") == group_id:
                alert_hit = g
                break
        for g in load_results("event_groups"):
            if g.get("group_id") == group_id:
                event_hit = g
                break
        return alert_hit, event_hit

    def _topk(self, query_vec, k: int = TOP_K) -> list[tuple[str, float]]:
        if self._matrix is None or query_vec is None:
            return []
        sims = (self._matrix @ query_vec.T).toarray().ravel()
        k = min(k, sims.shape[0])
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._group_ids[i], float(sims[i])) for i in top]

    # ── 核心：处理一条事件 ────────────────────────────────
    def process_event(self, payload: dict[str, Any], persist: bool = True) -> dict:
        self._ensure_ready()
        try:
            event = EventSchema.model_validate(payload)
        except Exception as exc:
            self.stats["error"] += 1
            return {"article_id": str(payload.get("article_file_id", "?")), "status": "error",
                    "message": f"字段校验失败: {exc}"}

        with self._lock:
            article_id = event.article_file_id
            owner = self._article_owner.get(article_id)
            if owner:
                self.stats["duplicate"] += 1
                return self._result(article_id, "duplicate", matched_group_id=owner,
                                    message="该文章已存在，未重复接入")

            query = None
            candidates: list[tuple[str, float]] = []
            if self._vectorizer is not None and self._matrix is not None:
                query = self._vectorizer.transform([_event_text(event)]).astype(np.float32)
                candidates = self._topk(query)
            result: dict

            if candidates and candidates[0][1] >= self.threshold:
                group_id, sim = candidates[0]
                result = self._assign_to_group(group_id, sim, event, query)
            else:
                result = self._create_group(event, query, candidates)

            result["candidates"] = [
                self._candidate_summary(gid, sim) for gid, sim in candidates
            ]
            if persist:
                self._append_log({"article_id": article_id, "payload": payload})
            return result

    def _assign_to_group(self, group_id: str, sim: float, event: EventSchema, query_vec) -> dict:
        alert_group, event_group = self._find_group(group_id)
        snapshot = _event_snapshot(event)
        # alerts 与 event_groups 缓存中可能是同一 dict 对象，按身份去重避免重复更新
        targets: list[dict] = []
        for g in (alert_group, event_group):
            if g is not None and not any(g is t for t in targets):
                targets.append(g)
        for g in targets:
            if g is None:
                continue
            g.setdefault("events", []).append(snapshot)
            g["article_count"] = (g.get("article_count") or 0) + 1
            sources = {e.get("article_source") for e in g["events"] if e.get("article_source")}
            g["source_diversity"] = round(len(sources) / len(g["events"]), 4)
            now_time = snapshot.get("article_publish_time")
            if now_time and (not g.get("timeline_end") or str(now_time) > str(g["timeline_end"])):
                g["timeline_end"] = now_time
            code = event.trading_code
            if code and code not in (g.get("trading_codes") or []):
                g.setdefault("trading_codes", []).append(code)
                g.setdefault("secu_names", []).append(event.secu_abbr or code)
        self._article_owner[event.article_file_id] = group_id

        alert: dict[str, Any] = {}
        message = "已归入既有事件组（评分沿用全量批处理结果）"
        if alert_group is not None:
            schema = _validated(alert_group)
            if schema is not None:
                if str(group_id).startswith("I"):
                    # 增量新建的组：用佐证启发式重算影响力，预警可随报道升级
                    new_impact = _impact_level_pair(alert_group.get("article_count") or 1,
                                                    alert_group.get("source_diversity") or 1.0)
                    schema = schema.model_copy(update={
                        "impact_score": new_impact[0],
                        "impact_level": new_impact[1],
                    })
                    scored = AlertEngine.apply(schema)
                    dumped = scored.model_dump(mode="json")
                    alert_group.update({
                        "impact_score": dumped["impact_score"],
                        "impact_level": dumped["impact_level"],
                        "alert_level": dumped["alert_level"],
                        "alert_reason": dumped["alert_reason"],
                        "alert_factors": dumped.get("alert_factors"),
                        "previous_alert_level": dumped.get("previous_alert_level"),
                        "alert_changed_reason": dumped.get("alert_changed_reason"),
                    })
                    alert = {
                        "level": scored.alert_level,
                        "reason": scored.alert_reason,
                        "changed_reason": scored.alert_changed_reason,
                    }
                    if scored.alert_changed_reason:
                        message = f"已归入事件组，预警{('升级' if (scored.alert_level or '') < (scored.previous_alert_level or '') else '降级')}：{scored.previous_alert_level} → {scored.alert_level}"
                    else:
                        message = "已归入事件组，预警维持"
                else:
                    alert = {
                        "level": schema.alert_level,
                        "reason": schema.alert_reason,
                        "changed_reason": schema.alert_changed_reason,
                    }
        self.stats["matched"] += 1
        return self._result(
            event.article_file_id, "matched",
            matched_group_id=group_id, similarity=sim, alert=alert,
            message="已归入既有事件组（评分沿用全量批处理结果）" if not alert.get("changed_reason") else message,
        )

    def _create_group(self, event: EventSchema, query_vec, candidates) -> dict:
        import scipy.sparse as sp

        self._new_group_seq += 1
        group_id = f"I{self._new_group_seq:06d}"
        snapshot = _event_snapshot(event)
        sources = {snapshot["article_source"]} if snapshot["article_source"] else set()
        diversity = round(len(sources) / 1, 4)
        ts = snapshot.get("article_publish_time") or datetime.now().isoformat()
        impact, impact_level = _impact_level_pair(1, diversity)

        group = {
            "group_id": group_id,
            "events": [snapshot],
            "cluster_confidence": 1.0,
            "timeline_start": ts,
            "timeline_end": ts,
            "merged_title": event.article_title or "(无标题)",
            "merged_content": (event.content or "")[:500],
            "article_count": 1,
            "source_diversity": diversity,
            "event_name": event.event_name or "未分类",
            "trading_codes": [event.trading_code] if event.trading_code else [],
            "secu_names": [event.secu_abbr] if event.secu_abbr else ([] if not event.trading_code else [event.trading_code]),
            "representative_article_id": event.article_file_id,
            "evidence_edges": [],
            "cluster_reason": "增量接入：低于相似度阈值，创建新事件组",
            "impact_score": impact,
            "credibility_score": 0.5,
            "sentiment_score": 0.0,
            "impact_level": impact_level,
            "b_score_components": {"ingest": "heuristic"},
            "b_score_reason": "增量接入启发式评分：报道佐证驱动",
        }
        schema = _validated(group)
        if schema is not None:
            scored = AlertEngine.apply(schema)
            group.update({k: scored.model_dump(mode="json").get(k) for k in
                          ("alert_level", "alert_reason", "alert_factors",
                           "previous_alert_level", "alert_changed_reason")})
            alert = {"level": scored.alert_level, "reason": scored.alert_reason,
                     "changed_reason": scored.alert_changed_reason}
        else:
            alert = {}

        load_results("alerts").append(group)
        load_results("event_groups").append(group)
        self._article_owner[event.article_file_id] = group_id
        # 新组向量并入索引，后续同类事件可命中
        if self._matrix is not None and query_vec is not None:
            self._matrix = sp.vstack([self._matrix, query_vec.astype(np.float32)]).tocsr()
            self._group_ids.append(group_id)
        self.stats["new"] += 1
        best_sim = candidates[0][1] if candidates else 0.0
        return self._result(
            event.article_file_id, "new",
            matched_group_id=group_id, similarity=best_sim, alert=alert,
            message=f"低于阈值 {self.threshold:.2f}，已创建新事件组 {group_id}",
        )

    # ── 结果/日志 ─────────────────────────────────────────
    def _candidate_summary(self, group_id: str, sim: float) -> dict:
        from src.api.routers.graph import _code_name_map

        g = next((x for x in load_results("alerts") if x.get("group_id") == group_id), None)
        if not g:
            return {"group_id": group_id, "similarity": round(sim, 4)}
        names = []
        authoritative = _code_name_map()
        for idx, code in enumerate(g.get("trading_codes") or []):
            raw = (g.get("secu_names") or [None] * (idx + 1))[idx]
            names.append(authoritative.get(str(code)) or raw or code)
        return {
            "group_id": group_id,
            "similarity": round(float(sim), 4),
            "title": (g.get("merged_title") or "")[:40],
            "event_name": g.get("event_name"),
            "companies": names[:3],
            "article_count": g.get("article_count"),
            "alert_level": g.get("alert_level", "P3"),
        }

    @staticmethod
    def _result(article_id: str, status: str, matched_group_id: str | None = None,
                similarity: float | None = None, alert: dict | None = None,
                message: str = "") -> dict:
        return {
            "article_id": article_id,
            "status": status,
            "matched_group_id": matched_group_id,
            "similarity": round(similarity, 4) if similarity is not None else None,
            "alert": alert or {},
            "message": message,
        }

    def _append_log(self, record: dict) -> None:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 持久化失败不影响当前请求

    def replay_log(self) -> int:
        """重启后重放接入日志，恢复缓存中的增量状态。"""
        if not LOG_PATH.exists():
            return 0
        replayed = 0
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    try:
                        self.process_event(record.get("payload") or {}, persist=False)
                        replayed += 1
                    except Exception:
                        continue
        except OSError:
            pass
        return replayed

    def reset(self) -> None:
        """清空增量状态：删日志、清缓存，并在后台重建索引（不阻塞请求）。

        重建 已有事件组的 TF-IDF 索引需要一定时间；若同步执行会阻塞请求约两分钟，
        因此改为后台线程重建，接口立即返回。重建期间 `/ingest` 会等待索引就绪。
        """
        with self._lock:
            self._ready = False
            self._index_error = None
            self._building = False
            self._group_ids = []
            self._article_owner = {}
            self._new_group_seq = 0
            self.stats = {"matched": 0, "new": 0, "duplicate": 0, "error": 0}
            if LOG_PATH.exists():
                LOG_PATH.unlink()
            clear_cache()
        threading.Thread(target=self.build_index, daemon=True, name="ingest-reset").start()


def find_group_in_caches(group_id: str) -> tuple[dict | None, dict | None]:
    """在 alerts 与 event_groups 两份缓存中查找组 dict（同一对象引用）。

    供类内方法与两阶段流水线共用。"""
    alert_hit = event_hit = None
    for g in load_results("alerts"):
        if g.get("group_id") == group_id:
            alert_hit = g
            break
    for g in load_results("event_groups"):
        if g.get("group_id") == group_id:
            event_hit = g
            break
    return alert_hit, event_hit


def _validated(group: dict) -> Any:
    """尽力把 dict 转 ScoredEventGroupSchema；失败返回 None。"""
    from src.api.schemas.scoring import ScoredEventGroupSchema

    try:
        return ScoredEventGroupSchema.model_validate(group)
    except Exception:
        return None


# 进程内单例
_service = IngestService()


def get_ingest_service() -> IngestService:
    return _service
