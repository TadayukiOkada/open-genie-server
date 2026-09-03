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


def order_slots(status: dict, at_least: int = 1, base_url: str = "") -> list:
    """Every loaded text slot as an (name, device_id) pair, same order.

    Which NSP core a slot sits on decides whether two of them can overlap, so a
    measurement that varies that has to see the device_id and not just the
    name. device_id is None for an unpinned slot: the model stays on whichever
    core its own HTP config names, which the server cannot read back.

    Unlike order_slot_names this returns ALL of them -- `at_least` is the
    minimum the caller needs, not how many it wants -- because a scaling
    measurement's whole point is to use however many are configured.
    """
    slots = [s for s in (status.get("slots") or []) if s.get("loaded")]
    slots.sort(key=lambda s: (s.get("device_id") is None, s.get("device_id")))
    if len(slots) < at_least:
        raise SystemExit(
            f"{base_url or 'the server'} has {len(slots)} loaded text slot(s) "
            f"({', '.join(s['name'] for s in slots) or 'none'}); this "
            f"measurement needs {at_least}. Point TEXT_SLOTS at the right "
            "configuration and restart.")
    return [(s["name"], s.get("device_id")) for s in slots]


def text_slots(base_url: str, at_least: int = 1, timeout: float = 10.0) -> list:
    r = requests.get(f"{base_url}/v1/server/status", timeout=timeout)
    r.raise_for_status()
    return order_slots(r.json(), at_least, base_url)


def same_core_pair(slots: list) -> list | None:
    """Two of `slots` (as returned by order_slots) sharing a device_id.

    Unpinned slots never qualify: device_id None means the model stayed on
    whichever core its own HTP config names, so the server cannot tell whether
    two of them landed together, and a measurement must not assume they did.
    """
    for i, (name, dev) in enumerate(slots):
        if dev is None:
            continue
        for other, other_dev in slots[i + 1:]:
            if other_dev == dev:
                return [name, other]
    return None


def cross_core_pair(slots: list) -> list | None:
    """The first slot plus the first one on a different device_id."""
    first, first_dev = slots[0]
    if first_dev is None:
        return None
    for name, dev in slots[1:]:
        if dev is not None and dev != first_dev:
            return [first, name]
    return None
