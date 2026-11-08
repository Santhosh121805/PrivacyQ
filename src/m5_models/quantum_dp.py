"""DP-SGD-capable quantum modules for Opacus.

Opacus computes per-sample gradients with per-layer "grad samplers". It has
none for a PennyLane circuit, so ``QuantumLayer`` registers one. Given the
layer input x (batch B) and the gradient of the loss w.r.t. the layer output,
the sampler re-runs the circuit with the weights replicated once per sample
(PennyLane broadcasts over that leading dimension) and takes one backward pass.
The gradient w.r.t. the i-th weight copy is exactly sample i's gradient
(verified against one backward pass per sample in tests). Opacus then does the
clipping, Gaussian noise and RDP accounting as for any other layer.

The circuit is a representation layer only; the privacy guarantee comes from
DP-SGD (clipping + noise + accounting), not from the quantum computation.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from opacus.grad_sample import register_grad_sampler

from src.m4_encoding import Encoding, make_encoding
from src.m5_models.quantum import DTYPE, _device


def _layers(encoding: Encoding, x, weights) -> None:
    """Variational circuit; ``weights`` may carry a leading per-sample batch dim."""
    n = encoding.n_qubits
    n_layers = weights.shape[-3]
    if encoding.reupload:
        for layer in range(n_layers):
            encoding.block(x)
            qml.StronglyEntanglingLayers(weights[..., layer:layer + 1, :, :], wires=range(n),
                                         ranges=[layer % (n - 1) + 1])
    else:
        encoding.feature_map(x)
        qml.StronglyEntanglingLayers(weights, wires=range(n))


class QuantumLayer(torch.nn.Module):
    """Encoding + StronglyEntanglingLayers; outputs <Z_w> for the readout wires."""

    def __init__(self, encoding: str, n_qubits: int, n_layers: int, readout: str = "first",
                 init_scale: float = 0.1):
        super().__init__()
        self.encoding = make_encoding(encoding, n_qubits)
        self.readout_wires = [0] if readout == "first" else list(range(n_qubits))
        self.weights = torch.nn.Parameter(init_scale * torch.randn(n_layers, n_qubits, 3, dtype=DTYPE))
        enc, wires = self.encoding, self.readout_wires

        @qml.qnode(_device(n_qubits), interface="torch", diff_method="backprop")
        def circuit(x, weights):
            _layers(enc, x, weights)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        self.circuit = circuit

    def run(self, x, weights) -> torch.Tensor:
        out = self.circuit(x, weights)
        return torch.stack(list(out), dim=-1) if isinstance(out, (list, tuple)) else out.unsqueeze(-1)

    def forward(self, x):
        if x.shape[0] == 0:
            return x.new_zeros((0, len(self.readout_wires))) + 0 * self.weights.sum()
        return self.run(x, self.weights)


@register_grad_sampler(QuantumLayer)
def _quantum_grad_sampler(layer: QuantumLayer, activations, backprops):
    x = activations[0] if isinstance(activations, (list, tuple)) else activations
    batch = x.shape[0]
    with torch.enable_grad():
        ghost = layer.weights.detach().unsqueeze(0).expand(batch, *layer.weights.shape).clone().requires_grad_(True)
        out = layer.run(x.detach(), ghost)
        (grad,) = torch.autograd.grad(out, ghost, grad_outputs=backprops.to(out.dtype))
    return {layer.weights: grad}


class AngleMap(torch.nn.Module):
    """Maps a compressed vector to rotation angles in [0, pi]."""

    def forward(self, z):
        return np.pi * torch.sigmoid(z)


class ConstantSlot(torch.nn.Module):
    """Appends the constant amplitude slot used by amplitude encoding."""

    def forward(self, z):
        return torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype)], dim=1)


def build_vqc(encoding: str, n_qubits: int, n_layers: int, init_scale: float = 0.1) -> torch.nn.Module:
    readout = torch.nn.Linear(1, 1, dtype=DTYPE)
    with torch.no_grad():  # logit = 1 * <Z_0> + 0 at initialisation, as in VQCClassifier
        readout.weight.fill_(1.0)
        readout.bias.zero_()
    return torch.nn.Sequential(QuantumLayer(encoding, n_qubits, n_layers, "first", init_scale), readout)


def build_hybrid(n_inputs: int, encoding: str, n_qubits: int, n_layers: int, init_scale: float = 0.1) -> torch.nn.Module:
    enc = make_encoding(encoding, n_qubits)
    width = enc.n_features(n_inputs)
    mapping = ConstantSlot() if enc.name == "amplitude" else AngleMap()
    return torch.nn.Sequential(
        torch.nn.Linear(n_inputs, width, dtype=DTYPE), mapping,
        QuantumLayer(encoding, n_qubits, n_layers, "all", init_scale),
        torch.nn.Linear(n_qubits, 1, dtype=DTYPE),
    )


def prepare_inputs(kind: str, encoding: str | None, n_qubits: int | None, X: np.ndarray) -> np.ndarray:
    if kind == "vqc":
        return make_encoding(encoding, n_qubits).prepare(X)
    return np.asarray(X, dtype=float)
