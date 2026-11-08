"""DP-SGD training through Opacus for every gradient-trained model.

- Opacus ``PrivacyEngine(accountant="rdp")`` with Poisson sampling, per-sample
  gradient clipping (``max_grad_norm``) and Gaussian noise calibrated by
  Opacus to reach ``target_epsilon`` at ``delta`` after ``epochs``.
- The reported epsilon is read back from that same Opacus accountant after
  training. ``epsilon=None`` (infinity) trains the same model with the same
  optimiser and no clipping or noise, and reports epsilon=None.
- Guarantee: (epsilon, delta)-DP for the trained weights with respect to adding
  or removing one training record (example-level DP). It covers the training
  rows passed to ``fit`` only. Not covered by the accounted epsilon:
  hyperparameter choices and the decision threshold / any selection that looks
  at the inner validation rows (documented in RESULTS.md).
- Opacus runs with ``secure_mode=False`` (the PyTorch RNG rather than a
  cryptographically secure one), as is standard for research experiments.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from opacus import PrivacyEngine
from sklearn.base import BaseEstimator, ClassifierMixin

from src.m5_models.classical import build_mlp, positive_weight
from src.m7_privacy.accounting import PrivacyReport, report_from_accountant

DP_MODELS = ("logreg", "mlp", "vqc", "hybrid")


def build_module(kind: str, n_inputs: int, params: dict) -> torch.nn.Module:
    if kind == "logreg":
        return torch.nn.Linear(n_inputs, 1, dtype=torch.float64)
    if kind == "mlp":
        return build_mlp(n_inputs, tuple(params.get("hidden", (64, 32)))).to(torch.float64)
    from src.m5_models.quantum_dp import build_hybrid, build_vqc

    if kind == "vqc":
        return build_vqc(params["encoding"], params["n_qubits"], params["n_layers"], params.get("init_scale", 0.1))
    if kind == "hybrid":
        return build_hybrid(n_inputs, params["encoding"], params["n_qubits"], params["n_layers"],
                            params.get("init_scale", 0.1))
    raise ValueError(f"unknown DP model {kind!r}")


class DPClassifier(BaseEstimator, ClassifierMixin):
    supports_eval_set = False

    def __init__(self, kind: str = "logreg", epsilon: float | None = None, delta: float = 1e-5,
                 max_grad_norm: float = 1.0, epochs: int = 30, batch_size: int = 128,
                 learning_rate: float = 0.05, seed: int = 0, model_params: dict | None = None):
        self.kind = kind
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_params = model_params

    def _inputs(self, X) -> torch.Tensor:
        params = self.model_params or {}
        if self.kind == "vqc":
            from src.m5_models.quantum_dp import prepare_inputs

            X = prepare_inputs("vqc", params["encoding"], params["n_qubits"], X)
        return torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.float64)

    def fit(self, X, y) -> "DPClassifier":
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        X_t = self._inputs(X)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float64)
        module = build_module(self.kind, X_t.shape[1], self.model_params or {})
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, y_t),
                                             batch_size=self.batch_size, shuffle=True, generator=generator)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight(y), dtype=torch.float64))
        engine = None
        if self.epsilon is not None:
            engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module, optimizer, loader = engine.make_private_with_epsilon(
                    module=module, optimizer=optimizer, data_loader=loader,
                    target_epsilon=float(self.epsilon), target_delta=float(self.delta),
                    epochs=int(self.epochs), max_grad_norm=float(self.max_grad_norm),
                    poisson_sampling=True,
                )
            self.noise_multiplier_ = float(optimizer.noise_multiplier)
        self.steps_ = 0
        module.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                if len(xb) == 0:  # Poisson sampling; probability ~(1-q)^n, negligible here
                    continue
                loss_fn(module(xb).reshape(-1), yb).backward()
                optimizer.step()
                self.steps_ += 1
        self.module_ = module
        if engine is None:
            self.privacy_ = PrivacyReport.not_private("non-private reference (epsilon = infinity)")
        else:
            self.privacy_ = report_from_accountant(engine.accountant, float(self.delta))
        self.classes_ = np.array([0, 1])
        return self

    def decision_scores(self, X) -> np.ndarray:
        self.module_.eval()
        with torch.no_grad():
            return self.module_(self._inputs(X)).reshape(-1).numpy()

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module_.parameters()))
