"""PennyLane feature encodings for small statevector circuits."""

from src.m4_encoding.amplitude import amplitude_encoding
from src.m4_encoding.angle import angle_encoding
from src.m4_encoding.zz_featuremap import zz_featuremap

ENCODINGS = {
    "angle": angle_encoding,
    "amplitude": amplitude_encoding,
    "zz": zz_featuremap,
}


def make_encoding(name: str, n_qubits: int, repetitions: int = 1):
    try:
        factory = ENCODINGS[name]
    except KeyError as error:
        raise ValueError(f"unknown encoding {name!r}") from error
    return factory(n_qubits=n_qubits, repetitions=repetitions)
