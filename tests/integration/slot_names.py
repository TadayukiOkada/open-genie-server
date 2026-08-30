"""Slot names for the measurement scripts, read from the running server.

The two-slot measurements used to hardcode cdsp0/cdsp1. Slot names are a
logical address, not hardware ("chat", "tool_call", "vision"), and the board's
own env_config.json is free to use either, so asking the server is the only
way these scripts keep working across a rename.

Order matters for reproducibility: results are reported per slot, so the
scripts need the same first slot on every run. Sorting by device_id keeps that
stable (cDSP0 before cDSP1) and puts unpinned slots last in configured order.
"""
import requests


def order_slot_names(status: dict, want: int = 1, base_url: str = "") -> list:
    """The names of `want` loaded text slots from a /v1/server/status body.

    Raises SystemExit with an actionable message when the server has fewer
    than `want` — the alternative is a stream of failures from every request.
    """
    slots = [s for s in (status.get("slots") or []) if s.get("loaded")]
    slots.sort(key=lambda s: (s.get("device_id") is None, s.get("device_id")))
    names = [s["name"] for s in slots]
    if len(names) < want:
        raise SystemExit(
            f"{base_url or 'the server'} has {len(names)} loaded text slot(s) "
            f"({', '.join(names) or 'none'}); this measurement needs {want}. "
            "Point TEXT_SLOTS at the right configuration and restart.")
    return names[:want]


def text_slot_names(base_url: str, want: int = 1, timeout: float = 10.0) -> list:
    r = requests.get(f"{base_url}/v1/server/status", timeout=timeout)
    r.raise_for_status()
    return order_slot_names(r.json(), want, base_url)
