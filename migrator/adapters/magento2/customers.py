"""Read Magento 2.x customers + default billing address into canonical Customer objects.

Unlike M1, M2's customer/address tables are flat (no EAV) — firstname, lastname,
email, etc. are direct columns on customer_entity and customer_address_entity.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Address, Customer


def load_customers(engine: Engine, prefix: str) -> list[Customer]:
    """prefix unused for M2 (no table prefix)."""
    with engine.connect() as conn:
        cust_rows = conn.execute(text(
            "SELECT entity_id, email, firstname, lastname, "
            "       created_at, default_billing "
            "FROM customer_entity "
            "WHERE email IS NOT NULL AND email <> '' "
            "ORDER BY entity_id"
        )).fetchall()

        newsletter_emails = {
            r.subscriber_email.lower()
            for r in conn.execute(text(
                "SELECT subscriber_email FROM newsletter_subscriber "
                "WHERE subscriber_status = 1 AND subscriber_email IS NOT NULL"
            )) if r.subscriber_email
        }
        addr_index = _load_addresses(conn)

    customers: list[Customer] = []
    for r in cust_rows:
        firstname = (r.firstname or "").strip()
        lastname = (r.lastname or "").strip()

        addr_id = r.default_billing
        try:
            addr_id = int(addr_id) if addr_id else None
        except (TypeError, ValueError):
            addr_id = None
        addr = addr_index.get(addr_id) if addr_id else None
        if addr:
            addr = Address(
                source_id=addr.source_id,
                customer_source_id=r.entity_id,
                firstname=addr.firstname or firstname,
                lastname=addr.lastname or lastname,
                company=addr.company,
                address_1=addr.address_1,
                address_2=addr.address_2,
                city=addr.city,
                postcode=addr.postcode,
                country_iso=addr.country_iso,
                zone_code=addr.zone_code,
                telephone=addr.telephone,
            )

        customers.append(Customer(
            source_id=r.entity_id,
            firstname=firstname,
            lastname=lastname,
            email=r.email,
            telephone=(addr.telephone if addr else ""),
            newsletter=(r.email.lower() in newsletter_emails),
            created_at=_parse_dt(r.created_at),
            address=addr,
        ))

    return customers


def _load_addresses(conn) -> dict[int, Address]:
    """customer_address_entity is flat in M2 — all fields are direct columns."""
    out: dict[int, Address] = {}
    rows = conn.execute(text(
        "SELECT entity_id, parent_id, firstname, lastname, company, "
        "       street, city, postcode, country_id, region, telephone "
        "FROM customer_address_entity"
    )).fetchall()
    for r in rows:
        street = r.street or ""
        lines = street.split("\n")
        out[r.entity_id] = Address(
            source_id=r.entity_id,
            customer_source_id=r.parent_id,
            firstname=(r.firstname or "").strip(),
            lastname=(r.lastname or "").strip(),
            company=(r.company or "").strip(),
            address_1=(lines[0] if lines else ""),
            address_2=(" ".join(lines[1:]) if len(lines) > 1 else ""),
            city=(r.city or "").strip(),
            postcode=(r.postcode or "").strip(),
            country_iso=(r.country_id or "").upper(),
            zone_code="",
            telephone=(r.telephone or "").strip(),
        )
    return out


def _parse_dt(v) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except Exception:
        return None
