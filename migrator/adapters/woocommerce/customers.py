"""Read WooCommerce customers (wp_users + wp_usermeta) into canonical Customer objects."""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Address, Customer

USER_META_KEYS = (
    "first_name", "last_name",
    "billing_first_name", "billing_last_name", "billing_company",
    "billing_address_1", "billing_address_2",
    "billing_city", "billing_state", "billing_postcode", "billing_country",
    "billing_phone", "billing_email",
)


def load_customers(engine: Engine, prefix: str) -> list[Customer]:
    with engine.connect() as conn:
        user_rows = conn.execute(text(
            f"SELECT ID, user_login, user_email, display_name, user_registered "
            f"FROM `{prefix}users` "
            f"WHERE user_email IS NOT NULL AND user_email <> '' "
            f"ORDER BY ID"
        )).fetchall()
        user_ids = [r.ID for r in user_rows]
        if not user_ids:
            return []

        meta = _load_meta(conn, prefix, user_ids)

    customers: list[Customer] = []
    for r in user_rows:
        m = meta.get(r.ID, {})
        firstname = (m.get("first_name") or m.get("billing_first_name") or "").strip()
        lastname = (m.get("last_name") or m.get("billing_last_name") or "").strip()
        if not firstname and not lastname and r.display_name:
            parts = r.display_name.strip().split(None, 1)
            firstname = parts[0] if parts else ""
            lastname = parts[1] if len(parts) > 1 else ""

        addr = None
        if m.get("billing_address_1") or m.get("billing_city") or m.get("billing_postcode"):
            addr = Address(
                source_id=r.ID,
                customer_source_id=r.ID,
                firstname=(m.get("billing_first_name") or firstname or "").strip(),
                lastname=(m.get("billing_last_name") or lastname or "").strip(),
                company=(m.get("billing_company") or "").strip(),
                address_1=(m.get("billing_address_1") or "").strip(),
                address_2=(m.get("billing_address_2") or "").strip(),
                city=(m.get("billing_city") or "").strip(),
                postcode=(m.get("billing_postcode") or "").strip(),
                country_iso=(m.get("billing_country") or "").strip().upper(),
                zone_code="",
                telephone=(m.get("billing_phone") or "").strip(),
            )

        customers.append(Customer(
            source_id=r.ID,
            firstname=firstname,
            lastname=lastname,
            email=r.user_email,
            telephone=(m.get("billing_phone") or "").strip(),
            newsletter=False,
            created_at=_parse_dt(r.user_registered),
            address=addr,
        ))

    return customers


def _load_meta(conn, prefix: str, ids: list[int]) -> dict[int, dict[str, str]]:
    placeholders = ",".join(str(i) for i in ids)
    key_list = ",".join(f"'{k}'" for k in USER_META_KEYS)
    out: dict[int, dict[str, str]] = {}
    for r in conn.execute(text(
        f"SELECT user_id, meta_key, meta_value FROM `{prefix}usermeta` "
        f"WHERE user_id IN ({placeholders}) AND meta_key IN ({key_list})"
    )):
        out.setdefault(r.user_id, {})[r.meta_key] = r.meta_value
    return out


def _parse_dt(v) -> Optional[datetime]:
    if not v or str(v).startswith("0000"):
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except Exception:
        return None
