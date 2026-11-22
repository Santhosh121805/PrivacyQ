"""Multiclass text models for the Mental Health Text dataset (box 10 baselines never trained on NSCH).

* ``MulticlassMLP``: feed-forward network on pooled features (the fourth classical baseline, multiclass).
* ``SequenceClassifier``: LSTM (bidirectional, max-pooled) or 1-D CNN (Kim-style, filter widths 3/4/5) over
  token ids. Token ids come from a frozen pretrained WordPiece tokenizer, so no vocabulary is fitted on the data;
  the embedding layer and the network are trained from scratch on each fold's inner-train rows only.

All models are class-weighted (inverse class frequency), expose ``class_scores(X)`` -> (n, k) probabilities and
``predict(X)``, and use early stopping on the inner-validation macro one-vs-rest ROC-AUC.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


def _macro_auc(y, P) -> float:
    k = P.shape[1]
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in range(k) if 0 < (y == c).sum() < len(y)]))


def _weights(y, k):
    return torch.tensor(compute_class_weight("balanced", classes=np.arange(k), y=y), dtype=torch.float32)


class _Torch(BaseEstimator, ClassifierMixin):
    supports_eval_set = True

    def predict(self, X):
        return np.argmax(self.class_scores(X), axis=1)

    def _train(self, net, batches, eval_fn, y, k):
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(weight=_weights(y, k))
        best, best_state, bad, self.history_ = -np.inf, None, 0, []
        for epoch in range(1, self.epochs + 1):
            net.train()
            for xb, yb in batches(epoch):
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            if eval_fn is None:
                continue
            auc = eval_fn()
            self.history_.append(auc)
            if auc > best:
                best, best_state, bad, self.best_epoch_ = auc, copy.deepcopy(net.state_dict()), 0, epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net


class MulticlassMLP(_Torch):
    def __init__(self, hidden: tuple[int, ...] = (256, 64), dropout: float = 0.2, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, epochs: int = 40, batch_size: int = 128, patience: int = 5, seed: int = 0):
        self.hidden, self.dropout, self.learning_rate = hidden, dropout, learning_rate
        self.weight_decay, self.epochs, self.batch_size, self.patience, self.seed = weight_decay, epochs, batch_size, patience, seed

    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.seed)
        X, y = torch.as_tensor(np.asarray(X), dtype=torch.float32), np.asarray(y)
        k = int(y.max()) + 1 if eval_set is None else int(max(y.max(), np.max(eval_set[1]))) + 1
        layers, d = [], X.shape[1]
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        self.net_ = torch.nn.Sequential(*layers, torch.nn.Linear(d, k))
        yt = torch.as_tensor(y, dtype=torch.long)
        g = torch.Generator().manual_seed(self.seed)

        def batches(_):
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(perm), self.batch_size):
                j = perm[i:i + self.batch_size]
                yield X[j], yt[j]
        ev = None if eval_set is None else (lambda: _macro_auc(np.asarray(eval_set[1]), self.class_scores(eval_set[0])))
        self.best_epoch_ = self.epochs
        self._train(self.net_, batches, ev, y, k)
        return self

    def class_scores(self, X):
        self.net_.eval()
        with torch.no_grad():
            return torch.softmax(self.net_(torch.as_tensor(np.asarray(X), dtype=torch.float32)), -1).numpy()

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net_.parameters()))


class _SeqNet(torch.nn.Module):
    def __init__(self, kind, vocab, emb, hidden, filters, k, dropout):
        super().__init__()
        self.kind = kind
        self.emb = torch.nn.Embedding(vocab, emb, padding_idx=0)
        self.drop = torch.nn.Dropout(dropout)
        if kind == "lstm":
            self.rnn = torch.nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
            out = 2 * hidden
        else:
            self.convs = torch.nn.ModuleList([torch.nn.Conv1d(emb, filters, w, padding=w // 2) for w in (3, 4, 5)])
            out = 3 * filters
        self.head = torch.nn.Linear(out, k)

    def forward(self, ids):
        mask = (ids != 0).unsqueeze(-1)
        e = self.drop(self.emb(ids))
        if self.kind == "lstm":
            h, _ = self.rnn(e)
            pooled = h.masked_fill(~mask, -1e9).max(dim=1).values
        else:
            x = e.transpose(1, 2)
            pooled = torch.cat([torch.relu(c(x))[:, :, :ids.shape[1]].masked_fill(~mask.transpose(1, 2), -1e9).max(dim=2).values
                                for c in self.convs], dim=1)
        return self.head(self.drop(pooled))


class SequenceClassifier(_Torch):
    """``X`` is an object array of int token-id arrays (already truncated); padding is per batch."""

    def __init__(self, kind: str = "lstm", vocab_size: int = 30522, emb_dim: int = 128, hidden: int = 96,
                 filters: int = 100, dropout: float = 0.3, learning_rate: float = 2e-3, weight_decay: float = 0.0,
                 epochs: int = 10, batch_size: int = 64, patience: int = 2, seed: int = 0):
        self.kind, self.vocab_size, self.emb_dim, self.hidden, self.filters = kind, vocab_size, emb_dim, hidden, filters
        self.dropout, self.learning_rate, self.weight_decay = dropout, learning_rate, weight_decay
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    @staticmethod
    def _pad(seqs):
        n = max(5, max(len(s) for s in seqs))
        out = np.zeros((len(seqs), n), dtype=np.int64)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return torch.as_tensor(out)

    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.seed)
        y = np.asarray(y)
        k = int(y.max()) + 1 if eval_set is None else int(max(y.max(), np.max(eval_set[1]))) + 1
        self.net_ = _SeqNet(self.kind, self.vocab_size, self.emb_dim, self.hidden, self.filters, k, self.dropout)
        lengths = np.array([len(s) for s in X])
        rng = np.random.default_rng(self.seed)
        yt = torch.as_tensor(y, dtype=torch.long)

        def batches(_):                                   # length-bucketed batches, shuffled order
            order = np.argsort(lengths + rng.uniform(0, 8, len(lengths)))
            chunks = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
            for j in rng.permutation(len(chunks)):
                idx = chunks[j]
                yield self._pad([X[i] for i in idx]), yt[idx]
        ev = None if eval_set is None else (lambda: _macro_auc(np.asarray(eval_set[1]), self.class_scores(eval_set[0])))
        self.best_epoch_ = self.epochs
        self._train(self.net_, batches, ev, y, k)
        return self

    def class_scores(self, X, batch: int = 256):
        self.net_.eval()
        order = np.argsort([len(s) for s in X])
        out = None
        with torch.no_grad():
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                p = torch.softmax(self.net_(self._pad([X[j] for j in idx])), -1).numpy()
                out = np.zeros((len(X), p.shape[1]), np.float32) if out is None else out
                out[idx] = p
        return out

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net_.parameters()))
