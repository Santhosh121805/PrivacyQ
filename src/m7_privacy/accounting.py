"""Privacy accounting. Opacus's RDP accountant is the ONLY source of epsilon.

There is deliberately no hand-written accountant in this project. A model whose
training is not DP-SGD run through Opacus (with its accountant tracking every
noisy step) reports ``epsilon = None``, never an estimate.

The guarantee is (epsilon, delta)-DP at the example (record) level for DP-SGD
with Poisson subsampling. It comes from gradient clipping plus Gaussian noise,
not from any quantum component.
"""

from __future__ import annotations

from dataclasses import dataclass

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


@dataclass(frozen=True)
class PrivacyReport:
    dp_trained: bool
    epsilon: float | None
    delta: float | None
    noise_multiplier: float | None = None
    sample_rate: float | None = None
    steps: int | None = None
    accountant: str | None = None
    note: str = ""

    @classmethod
    def not_private(cls, reason: str) -> "PrivacyReport":
        return cls(dp_trained=False, epsilon=None, delta=None, note=reason)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def epsilon_for(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """Epsilon spent by ``steps`` DP-SGD steps, from Opacus's RDP accountant."""
    accountant = RDPAccountant()
    accountant.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
    return float(accountant.get_epsilon(delta=float(delta)))


def noise_for(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """Noise multiplier achieving ``target_epsilon`` (Opacus's own search)."""
    return float(
        get_noise_multiplier(
            target_epsilon=float(target_epsilon), target_delta=float(delta),
            sample_rate=float(sample_rate), steps=int(steps), accountant="rdp",
        )
    )


def report_from_accountant(accountant: RDPAccountant, delta: float) -> PrivacyReport:
    """Build a report from the accountant that tracked an actual Opacus training run."""
    if not accountant.history:
        return PrivacyReport.not_private("accountant recorded no DP-SGD steps")
    noise, rate, _ = accountant.history[-1]
    steps = sum(step for _, _, step in accountant.history)
    return PrivacyReport(
        dp_trained=True, epsilon=float(accountant.get_epsilon(delta=delta)), delta=delta,
        noise_multiplier=float(noise), sample_rate=float(rate), steps=int(steps), accountant="opacus.rdp",
    )
