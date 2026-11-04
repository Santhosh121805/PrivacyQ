"""Secure aggregation is NOT implemented. This stub only reports that fact."""

from __future__ import annotations


def secure_aggregation_stub(client_updates: list | None = None) -> dict[str, object]:
    payload = {"status": "stub", "message": "secure aggregation is not implemented"}
    if client_updates is not None:
        payload["n_clients"] = len(client_updates)
    return payload
