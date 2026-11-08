"""PennyLane feature encodings (representation only; no privacy role)."""

from src.m4_encoding.encodings import (
    ENCODING_NAMES,
    Encoding,
    circuit_resources,
    encoding_resources,
    make_encoding,
)

__all__ = ["ENCODING_NAMES", "Encoding", "circuit_resources", "encoding_resources", "make_encoding"]
