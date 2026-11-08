"""字符 n-gram TF-IDF 检索（自带实现，仅依赖 numpy + scipy）。

用于轻量增量归组的向量化与相似度检索。

为什么不用 `sklearn.feature_extraction.text.TfidfVectorizer`：
  其 `char_wb` 分析器在部分环境/版本下会抛出与 n-gram 步长相关的异常
  （`TypeError: unsupported operand type(s) for +: 'range_iterator' and 'int'`），
  导致轻量接入与自然语言接入直接不可用。自带实现行为可控、结果可测，
  同时把 sklearn 从 Web 运行时依赖中移除。

实现要点：
  - 字符 n-gram（默认 2–3 gram），空白归一为单空格；
  - TF-IDF：sublinear_tf（1+ln tf）× idf（ln((1+N)/(1+df))+1），按文档 L2 归一化；
  - min_df 过滤长尾、max_features 截断词表；
  - 输出 scipy CSR，查询用稀疏点积（归一化后即余弦相似度）。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

_WS = re.compile(r"\s+")


class CharNgramTfidf:
    """字符 n-gram TF-IDF 向量化器（sklearn 兼容的最小接口）。"""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 3),
        min_df: int = 2,
        max_features: int | None = 150_000,
        sublinear_tf: bool = True,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.n_docs = 0

    # ── 文本 → gram 计数 ──────────────────────────────────
    @staticmethod
    def _normalize(text: str) -> str:
        return _WS.sub(" ", str(text or "")).strip()

    def _grams(self, text: str) -> Counter[str]:
        s = self._normalize(text)
        out: Counter[str] = Counter()
        if not s:
            return out
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if len(s) < n:
                out[s] += 1
                continue
            for i in range(len(s) - n + 1):
                out[s[i : i + n]] += 1
        return out

    # ── 拟合 ─────────────────────────────────────────────
    def fit(self, texts: Sequence[str]) -> "CharNgramTfidf":
        df: Counter[str] = Counter()
        for text in texts:
            df.update(self._grams(text).keys())  # 每篇每 gram 只计一次

        items = [(g, c) for g, c in df.items() if c >= self.min_df]
        items.sort(key=lambda kv: (-kv[1], kv[0]))  # 高频优先，稳定排序
        if self.max_features:
            items = items[: self.max_features]

        self.vocabulary_ = {gram: idx for idx, (gram, _) in enumerate(items)}
        self.n_docs = len(texts)
        idf = np.ones(len(self.vocabulary_), dtype=np.float32)
        for gram, idx in self.vocabulary_.items():
            idf[idx] = math.log((1.0 + self.n_docs) / (1.0 + df[gram])) + 1.0
        self.idf_ = idf
        return self

    def fit_transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        self.fit(texts)
        return self.transform(texts)

    # ── 变换 ─────────────────────────────────────────────
    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("CharNgramTfidf 尚未 fit")
        vocab = self.vocabulary_
        idf = self.idf_
        indptr: list[int] = [0]
        indices: list[int] = []
        data: list[float] = []
        n_rows = 0

        for text in texts:
            n_rows += 1
            pairs: list[tuple[int, float]] = []
            for gram, count in self._grams(text).items():
                idx = vocab.get(gram)
                if idx is None:
                    continue
                tf = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
                pairs.append((idx, tf * float(idf[idx])))
            pairs.sort()
            if pairs:
                idxs = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
                vals = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
                norm = float(np.linalg.norm(vals))
                if norm > 0:
                    vals = vals / norm
                indices.extend(idxs.tolist())
                data.extend(vals.tolist())
            indptr.append(len(indices))

        return sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(n_rows, len(vocab)),
        )

    # ── 兼容 sklearn 的属性名 ─────────────────────────────
    @property
    def vocabulary(self) -> dict[str, int]:  # pragma: no cover - 兼容别名
        return self.vocabulary_
