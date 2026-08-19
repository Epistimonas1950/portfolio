"""Shared EAV helpers for the Magento 2 adapter.

M2 attribute IDs are auto-incremented per install — they cannot be hardcoded
like we did for M1. Each entity type's attribute set must be looked up via
`eav_attribute` joined to `eav_entity_type`.
"""
from __future__ import annotations
from typing import Optional

from sqlalchemy import text


def load_attribute_map(conn, entity_type_code: str) -> dict[str, tuple[int, str]]:
    """Return {attribute_code: (attribute_id, backend_type)} for an entity type.

    backend_type is one of: varchar, int, text, decimal, datetime, static.
    """
    rows = conn.execute(text(
        "SELECT a.attribute_id, a.attribute_code, a.backend_type "
        "FROM eav_attribute a "
        "JOIN eav_entity_type t ON t.entity_type_id = a.entity_type_id "
        "WHERE t.entity_type_code = :code"
    ), {"code": entity_type_code}).fetchall()
    return {r.attribute_code: (r.attribute_id, r.backend_type) for r in rows}


def attrs_by_backend(
    attr_map: dict[str, tuple[int, str]],
    wanted_codes: list[str],
) -> dict[str, dict[int, str]]:
    """Group the requested attribute codes by backend_type table.

    Returns {backend_type: {attribute_id: attribute_code}}.
    Codes not present in attr_map are silently skipped.
    """
    out: dict[str, dict[int, str]] = {}
    for code in wanted_codes:
        if code not in attr_map:
            continue
        aid, btype = attr_map[code]
        out.setdefault(btype, {})[aid] = code
    return out


def get_entity_type_id(conn, entity_type_code: str) -> Optional[int]:
    row = conn.execute(text(
        "SELECT entity_type_id FROM eav_entity_type WHERE entity_type_code = :c"
    ), {"c": entity_type_code}).fetchone()
    return row.entity_type_id if row else None
