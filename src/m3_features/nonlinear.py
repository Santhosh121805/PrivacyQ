"""Non-linear dimensionality reduction: UMAP and a small autoencoder.

Both are unsupervised (labels are not used, as for PCA) and are fitted on the training rows passed
to ``fit`` only; ``transform`` maps any later rows with the fitted reducer.
"""

from __future__ import annotations

import numpy as np
import torch


class UMAPReducer:
    """UMAP (McInnes et al.) to ``n_components`` dimensions; ``transform`` embeds new rows."""

    def __init__(self, n_components: int, seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.1):
        self.n_components = n_components
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import umap

        self.model_ = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                                min_dist=self.min_dist, random_state=self.seed, n_jobs=1)
        return np.asarray(self.model_.fit_transform(X), dtype=float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.transform(X), dtype=float)


class AutoencoderReducer:
    """Fully connected autoencoder input -> 32 -> n_components -> 32 -> input (ReLU hidden layers,
    linear bottleneck), trained with Adam on mean squared reconstruction error. The bottleneck
    activations are the reduced features."""

    def __init__(self, n_components: int, seed: int = 0, hidden: int = 32, epochs: int = 100,
                 batch_size: int = 128, learning_rate: float = 1e-3):
        self.n_components = n_components
        self.seed = seed
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        torch.manual_seed(self.seed)
        d = X.shape[1]
        self.encoder_ = torch.nn.Sequential(torch.nn.Linear(d, self.hidden), torch.nn.ReLU(),
                                            torch.nn.Linear(self.hidden, self.n_components)).double()
        decoder = torch.nn.Sequential(torch.nn.Linear(self.n_components, self.hidden), torch.nn.ReLU(),
                                      torch.nn.Linear(self.hidden, d)).double()
        params = list(self.encoder_.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.learning_rate)
        X_t = torch.as_tensor(X, dtype=torch.float64)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t), batch_size=self.batch_size,
                                             shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        self.loss_history_ = []
        for _ in range(self.epochs):
            total = 0.0
            for (xb,) in loader:
                opt.zero_grad()
                loss = torch.mean((decoder(self.encoder_(xb)) - xb) ** 2)
                loss.backward()
                opt.step()
                total += float(loss) * len(xb)
            self.loss_history_.append(total / len(X_t))
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.encoder_(torch.as_tensor(X, dtype=torch.float64)).numpy()
