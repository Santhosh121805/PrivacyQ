"""Text features from frozen pretrained models (box 3, Mental Health Text dataset).

No model here is trained or fine-tuned: each text is mapped independently by fixed pretrained weights, so the
features can be computed once for every row without fitting anything on the data. Anything that *is* fitted
(mutual-information ranking, scalers, PCA later) uses development rows only.

  * pooled sentence embedding of the whole text (truncated to ``max_tokens``)
  * per-sentence embeddings pooled by mean and by max over the text's sentences (no truncation of the text)
  * emotion probabilities from a frozen emotion classifier (truncated to ``max_tokens``)
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, min_chars: int = 2) -> list[str]:
    """Rule-based sentence split on . ! ? and line breaks; a text with no boundary is one sentence."""
    parts = [p.strip() for p in _SENT.split(str(text))]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts or [str(text).strip()]


def token_lengths(tokenizer, texts: Sequence[str], batch: int = 2048) -> np.ndarray:
    """Untruncated token count per text (special tokens included) under ``tokenizer``."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(list(texts[i:i + batch]), add_special_tokens=True, truncation=False)["input_ids"]
        out.extend(len(e) for e in enc)
    return np.asarray(out)


def pooled_embeddings(model, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """One vector per text from a frozen SentenceTransformer (its own pooling and max_seq_length)."""
    return np.asarray(model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                                   convert_to_numpy=True), dtype=np.float32)


def sentence_pooled_embeddings(model, texts: Sequence[str], batch_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed every sentence of every text; return (mean over sentences, max over sentences, sentences per text)."""
    sents, owner = [], []
    for k, t in enumerate(texts):
        s = split_sentences(t)
        sents.extend(s)
        owner.extend([k] * len(s))
    owner = np.asarray(owner)
    uniq, inv = np.unique(np.asarray(sents, dtype=object), return_inverse=True)
    E = pooled_embeddings(model, uniq.tolist(), batch_size=batch_size)[inv]
    n = np.bincount(owner, minlength=len(texts))
    mean = np.zeros((len(texts), E.shape[1]), np.float32)
    np.add.at(mean, owner, E)
    mean /= n[:, None]
    mx = np.full((len(texts), E.shape[1]), -np.inf, np.float32)
    np.maximum.at(mx, owner, E)
    return mean, mx, n


def emotion_probabilities(tokenizer, model, texts: Sequence[str], max_tokens: int = 256, batch_size: int = 32,
                          progress=None) -> np.ndarray:
    """Softmax probabilities of a frozen sequence classifier; texts processed in length order to cut padding."""
    import torch

    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), np.float32)
    with torch.no_grad():
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_tokens, padding=True,
                            return_tensors="pt")
            out[idx] = torch.softmax(model(**enc).logits, dim=-1).numpy()
            if progress is not None and (i // batch_size) % 50 == 0:
                progress(i + len(idx), len(order))
    return out


def mi_ranking(X: np.ndarray, y: np.ndarray, columns: Sequence[str], labels: Sequence[str], seed: int,
               top: int = 8) -> "pd.DataFrame":  # noqa: F821
    """Mutual information of each column with the label, in results/feature_importance/mi_scores.csv format.

    Call with development rows only; the hold-out never enters a ranking.
    """
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(X, y, random_state=seed)
    out = pd.DataFrame({"column": list(columns), "label": list(labels), "mutual_information": scores})
    out = out.sort_values("mutual_information", ascending=False, kind="stable").reset_index(drop=True)
    out["selected_top8"] = out.index < top
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["column", "label", "mutual_information", "selected_top8", "rank"]]
