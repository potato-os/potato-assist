"""Live and Static Context grouped by area, for models trained on it (lfm3-1.7b-home-assistant).

Ported from the training pipeline's reference implementation (26 Oct 2026, sha256 7d92084c...); the model was trained
and scored on exactly that code's output, so keep these functions byte-for-byte equivalent to it. A 2B model does not read a missing area tag reliably (asked for the
bedroom temperature it also reported the no-area dehumidifier), so scope is a matter of which block a line sits in:

    Live Context: An overview of the areas and the devices in this smart home:
    Devices are grouped by area. Devices under "No area" are not in any area.
    Bedroom:
    - CO2 Monitor Temperature (sensor): 23 °C
    No area:
    - Dehumidifier Temperature (sensor): 21.9 °C

HA's YAML is parsed with yaml.safe_load (HA wraps long values and quotes 'on'/'Yes'/'012'). One block per distinct
`areas` string, used verbatim (area name plus aliases; never split), sorted case-insensitively; entities without an
area last under "No area:", and a real area called "No area" gets "No area (area):". HA's order is kept inside a
block. Anything that does not parse as a list of entities, or a result with "error" true, is left unchanged.
"""

import json

import yaml

STATIC_HEAD = "Static Context: An overview of the areas and the devices in this smart home:"
NO_AREA = "No area"
SCOPE_LINE = f'Devices are grouped by area. Devices under "{NO_AREA}" are not in any area.'


def _state_text(it):
    """State, unit and attributes as the v1 compact lines had them; device_class and state_class dropped."""
    st = it.get("state")
    attrs = dict(it.get("attributes") or {})
    unit = attrs.pop("unit_of_measurement", None)
    attrs.pop("device_class", None)
    attrs.pop("state_class", None)
    s = "" if st is None else str(st)
    if s in ("unavailable", "unknown"):
        unit, attrs = None, {}
    if unit and s:
        s += " " + str(unit)
    extra = ", ".join(f"{k} {v}" for k, v in attrs.items() if v not in (None, ""))
    return s + (f"; {extra}" if extra else "")


def _header(area):
    if area is None:
        return f"{NO_AREA}:"
    return f"{NO_AREA} (area):" if area.strip().casefold() == NO_AREA.casefold() else f"{area}:"


def _blocks(entries):
    """entries: (areas string or None, line) in HA's order -> header + lines per block."""
    by = {}
    for area, line in entries:
        by.setdefault(area if area else None, []).append(line)
    out = []
    for area in sorted((a for a in by if a is not None), key=lambda a: (a.casefold(), a)) + ([None] if None in by else []):
        out.append(_header(area))
        out += by[area]
    return out


def _items(body):
    try:
        items = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    if not isinstance(items, list) or not items or not all(isinstance(it, dict) and "names" in it for it in items):
        return None
    return items


def group_live_text(text):
    """HA's live-context text (header line + YAML) -> grouped text, or None when it is not that."""
    head, _, body = text.partition("\n")
    if not head.endswith(":"):
        return None
    items = _items(body)
    if items is None:
        return None
    entries = []
    for it in items:
        st = _state_text(it)
        entries.append((it.get("areas"), f"- {it['names']} ({it.get('domain')}): {st}".rstrip()))
    return "\n".join([head, SCOPE_LINE] + _blocks(entries))


def rewrite_tool_content(content):
    """A GetLiveContext tool message ({"data":{"result":...},"error":false}) -> grouped text; anything else unchanged."""
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return content
    if not isinstance(obj, dict) or obj.get("error"):
        return content
    data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
    res = data.get("result") if isinstance(data, dict) else None
    if not isinstance(res, str):
        return content
    out = group_live_text(res)
    return content if out is None else out


def rewrite_system(system):
    """HA's system prompt with its YAML Static Context (header up to the next blank line) -> grouped blocks."""
    i = system.find(STATIC_HEAD)
    if i < 0:
        return system
    j = system.find("\n\n", i)
    j = len(system) if j < 0 else j
    items = _items(system[i + len(STATIC_HEAD):j])
    if items is None:
        return system
    entries = [(it.get("areas"), f"- {it['names']} ({it.get('domain')})") for it in items]
    return system[:i] + "\n".join([STATIC_HEAD, SCOPE_LINE] + _blocks(entries)) + system[j:]
