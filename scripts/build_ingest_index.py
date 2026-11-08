"""构建轻量归组的字符 n-gram TF-IDF 索引（**在独立进程中执行**）。

为什么不直接在 API 进程内构建：
  长时间、高分配量的纯 Python 循环在个别环境下会间歇性触发段错误
  （本项目历史上记录过同类“解释器/原生库”问题）。把构建放到子进程后，
  即使子进程崩溃，API 主进程仍可正常服务，并退化为“仅新建事件组”。

输出（默认 data/cache/ingest_index.npz）：
  vocab        词表（按 id 排序的 gram 列表）
  idf          idf 权重
  csr_data / csr_indices / csr_indptr / shape
  group_ids    与矩阵行一一对应的事件组 ID
  source_mtime 构建时 alerts 轻量结果文件的 mtime（用于判断缓存是否新鲜）

用法：
    python -X utf8 scripts/build_ingest_index.py [输出路径]
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.data_store import _resolve  # noqa: E402
from src.api.data_store import load_results  # noqa: E402
from src.services.char_tfidf import CharNgramTfidf  # noqa: E402
from src.services.ingest_service import _group_text  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "cache" / "ingest_index.npz"


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    groups = load_results("alerts")
    group_ids = [g.get("group_id") for g in groups]
    texts = [_group_text(g) for g in groups]
    print(f"[index] 载入 {len(groups):,} 组（{time.time() - started:.1f}s）", flush=True)

    vectorizer = CharNgramTfidf(
        ngram_range=(2, 3), min_df=2, max_features=150_000, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts).astype(np.float32).tocsr()
    del texts
    gc.collect()
    print(f"[index] 索引完成 词表={len(vectorizer.vocabulary_):,} nnz={matrix.nnz:,}"
          f"（{time.time() - started:.1f}s）", flush=True)

    vocab = [None] * len(vectorizer.vocabulary_)
    for gram, idx in vectorizer.vocabulary_.items():
        vocab[idx] = gram

    source = _resolve("alerts")
    np.savez_compressed(
        out_path,
        vocab=np.asarray(vocab, dtype=object),
        idf=vectorizer.idf_.astype(np.float32),
        csr_data=matrix.data,
        csr_indices=matrix.indices,
        csr_indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        group_ids=np.asarray(group_ids, dtype=object),
        source_mtime=np.asarray([source.stat().st_mtime if source.exists() else 0.0]),
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[index] 已写入 {out_path}（{size_mb:.1f}MB，总耗时 {time.time() - started:.1f}s）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
