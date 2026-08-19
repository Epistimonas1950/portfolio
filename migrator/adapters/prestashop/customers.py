"""Read PrestaShop customers + addresses into canonical Customer objects."""
from __future__ import annotations
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Address, Customer
from migrator.adapters.prestashop._zones import ps_zone_to_oc


def load_customers(engine: Engine, prefix: str) -> list[Customer]:
    with engine.connect() as conn:
        addr_idx = _load_addresses(conn, prefix)
        rows = conn.execute(text(
            f"SELECT id_customer, firstname, lastname, email, newsletter, date_add "
            f"FROM `{prefix}customer` "
            f"WHERE active = 1 AND deleted = 0 "
            f"ORDER BY id_customer"
        )).fetchall()

    out: list[Customer] = []
    for r in rows:
        addr = addr_idx.get(r.id_customer)
        out.append(Customer(
            source_id=r.id_customer,
            firstname=(r.firstname or "").strip(),
            lastname=(r.lastname or "").strip(),
            email=(r.email or "").strip().lower(),
            telephone=addr.telephone if addr else "",
            newsletter=bool(r.newsletter),
            created_at=r.date_add,
            address=addr,
        ))
    return out


def _load_addresses(conn, prefix: str) -> dict[int, Address]:
    """Return {customer_source_id: Address} — most recent non-deleted address per customer."""
    rows = conn.execute(text(
        f"SELECT a.id_address, a.id_customer, a.firstname, a.lastname, a.company, "
        f"a.address1, a.address2, a.postcode, a.city, "
        f"a.phone, a.phone_mobile, "
        f"c.iso_code AS country_iso, s.iso_code AS zone_iso "
        f"FROM `{prefix}address` a "
        f"LEFT JOIN `{prefix}country` c ON c.id_country = a.id_country "
        f"LEFT JOIN `{prefix}state` s ON s.id_state = a.id_state "
        f"WHERE a.deleted = 0 AND a.id_customer > 0 "
        f"ORDER BY a.id_customer, a.date_upd DESC"
    )).fetchall()

    out: dict[int, Address] = {}
    for r in rows:
        if r.id_customer in out:
            continue
        phone = (r.phone_mobile or r.phone or "").strip()
        country_iso = ((r.country_iso or "")[:2]).upper()
        out[r.id_customer] = Address(
            source_id=r.id_address,
            customer_source_id=r.id_customer,
            firstname=(r.firstname or "").strip(),
            lastname=(r.lastname or "").strip(),
            company=(r.company or "").strip(),
            address_1=(r.address1 or "").strip(),
            address_2=(r.address2 or "").strip(),
            city=(r.city or "").strip(),
            postcode=(r.postcode or "").strip(),
            country_iso=country_iso,
            zone_code=ps_zone_to_oc(country_iso, r.zone_iso or ""),
            telephone=phone,
        )
    return out
