"""构建「事件组代表向量」索引（方案 A 前置：让新事件能归入存量 已有事件组）。

一次性任务（约 5 分钟，GPU）：
  1. 读 data/web/alerts.json.gz 的全部事件组；
  2. 取每组代表文本（merged_title×2 + event_name + 公司简称 + merged_content 摘要）；
  3. BGE 嵌入 → 归一化 → 保存：
       data/embeddings/group_reps.npy    (N × 1024, float32)
       data/embeddings/group_ids.json    (并行数组: group_id 列表)
       data/embeddings/group_meta.json   (并行数组: 标题/事件名/公司 摘要, 供前端展示)

用法: .venv/bin/python scripts/build_group_index.py
"""
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.api.data_store import load_results
from src.config import settings
from src.features.embedding import EmbeddingEngine

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "embeddings"
BATCH = 512


def group_text(group: dict) -> str:
    title = group.get("merged_title") or ""
    content = (group.get("merged_content") or "")[:200]
    secu = " ".join(group.get("secu_names") or [])
    return " ".join(p for p in (title, title, group.get("event_name") or "", secu, content) if p)


def main() -> None:
    groups = load_results("alerts")
    print(f"事件组总数: {len(groups):,}")
    texts = [group_text(g) for g in groups]

    embedder = EmbeddingEngine()
    t0 = time.time()
    vecs = embedder.encode(texts, show_progress=True)
    print(f"嵌入完成: {vecs.shape}，耗时 {time.time() - t0:.1f}s")

    # 归一化（余弦 == 内积）
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT_DIR / "group_reps.npy"), vecs)
    with open(OUT_DIR / "group_ids.json", "w", encoding="utf-8") as f:
        json.dump([g.get("group_id") for g in groups], f, ensure_ascii=False)

    meta = []
    for g in groups:
        meta.append({
            "title": (g.get("merged_title") or "")[:60],
            "event_name": g.get("event_name"),
            "companies": (g.get("secu_names") or [])[:3],
            "article_count": g.get("article_count"),
            "alert_level": g.get("alert_level", "P3"),
        })
    with open(OUT_DIR / "group_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"完成: {OUT_DIR / 'group_reps.npy'} ({vecs.nbytes / 1024 / 1024:.1f}MB)")


if __name__ == "__main__":
    main()
