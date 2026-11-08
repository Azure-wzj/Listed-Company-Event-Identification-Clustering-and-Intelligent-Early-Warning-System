"""存量事件组的代表向量索引 —— 新事件可"归进已聚好的 已有事件组"。

索引由 scripts/build_group_index.py 一次性生成（BGE，约 40 分钟）：
  data/embeddings/group_reps.npy   (N × 1024, float32, 已归一化)
  data/embeddings/group_ids.json   (并行: group_id)
  data/embeddings/group_meta.json  (并行: 标题/事件名/公司/报道数/预警 摘要)

查询：新事件向量 → 余弦相似度（=内积，已归一化）→ TopK + 阈值。
mmap 只读加载：297MB 由 OS 页缓存管理，不占 Python 堆内存
（WSL 默认内存 7.6GB，段错误根因之一是堆内存不足）。
无需 FAISS：N=已有 × 1024 的矩阵乘在 GPU/CPU 上毫秒级。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPS_PATH = ROOT / "data" / "embeddings" / "group_reps.npy"
IDS_PATH = ROOT / "data" / "embeddings" / "group_ids.json"
META_PATH = ROOT / "data" / "embeddings" / "group_meta.json"

# 标题级代表向量的归组阈值（BGE 量纲）。离线校准见 docs；可在测试后调整。
DEFAULT_THRESHOLD = 0.82


class GroupVectorIndex:
    """线程安全的存量组向量索引（mmap 加载）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._matrix: Optional[np.ndarray] = None
        self._group_ids: list[str] = []
        self._meta: list[dict] = []
        self._loaded = False
        self._error: str | None = None

    def load(self) -> bool:
        with self._lock:
            if self._loaded:
                return True
            try:
                if not REPS_PATH.exists():
                    self._error = f"索引缺失: {REPS_PATH}（先运行 scripts/build_group_index.py）"
                    return False
                # mmap 只读：297MB 由 OS 页缓存兜底，不占堆内存（WSL 内存受限）
                self._matrix = np.load(str(REPS_PATH), allow_pickle=False, mmap_mode="r").astype(np.float32, copy=False)
                with open(IDS_PATH, "r", encoding="utf-8") as f:
                    ids = json.load(f)
                self._group_ids = list(ids)
                self._meta = []
                if META_PATH.exists():
                    with open(META_PATH, "r", encoding="utf-8") as f:
                        self._meta = list(json.load(f))
                if self._matrix.shape[0] != len(self._group_ids):
                    self._error = f"索引与 ID 数量不一致: {self._matrix.shape[0]} != {len(self._group_ids)}"
                    return False
                self._loaded = True
                self._error = None
                return True
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                return False

    @property
    def ready(self) -> bool:
        return self._loaded

    @property
    def error(self) -> str | None:
        return self._error

    def status(self) -> dict:
        return {
            "ready": self._loaded,
            "error": self._error,
            "groups": len(self._group_ids),
            "dim": int(self._matrix.shape[1]) if self._matrix is not None else 0,
            "threshold": DEFAULT_THRESHOLD,
        }

    def topk(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """query: (dim,) 已归一化向量 → [(位置, 相似度)] 降序。矩阵乘在页缓存上完成。"""
        if not self._loaded or self._matrix is None:
            return []
        sims = self._matrix @ query  # (N,)
        k = min(k, len(sims))
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def candidate_summary(self, index: int) -> dict:
        if 0 <= index < len(self._meta):
            return dict(self._meta[index])
        return {"group_id": self._group_ids[index] if 0 <= index < len(self._group_ids) else None}

    @property
    def group_ids(self) -> list[str]:
        return self._group_ids


_index_singleton = None
_index_singleton_lock = threading.Lock()


def get_group_index() -> GroupVectorIndex:
    global _index_singleton
    with _index_singleton_lock:
        if _index_singleton is None:
            _index_singleton = GroupVectorIndex()
        return _index_singleton
