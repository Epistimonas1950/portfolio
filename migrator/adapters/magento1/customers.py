"""Read Magento 1.x customers + default billing address into canonical Customer objects."""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Address, Customer

CUST_ATTR_VARCHAR = {1: "firstname", 2: "lastname"}
CUST_ATTR_INT     = {7: "default_billing", 8: "default_shipping"}

ADDR_ATTR_VARCHAR = {9: "firstname", 10: "lastname", 11: "country_id",
                     12: "region", 14: "postcode", 15: "city",
                     17: "telephone", 18: "fax", 95: "company"}
ADDR_ATTR_TEXT    = {16: "street"}
ADDR_ATTR_INT     = {13: "region_id"}


def load_customers(engine: Engine, prefix: str) -> list[Customer]:
    with engine.connect() as conn:
        cust_rows = conn.execute(text(
            "SELECT entity_id, email, created_at, website_id, store_id, group_id "
            "FROM customer_entity WHERE email IS NOT NULL AND email <> '' "
            "ORDER BY entity_id"
        )).fetchall()

        cust_eav = _load_customer_eav(conn)
        newsletter_emails = {
            r.subscriber_email.lower()
            for r in conn.execute(text(
                "SELECT subscriber_email FROM newsletter_subscriber "
                "WHERE subscriber_status = 1 AND subscriber_email IS NOT NULL"
            )) if r.subscriber_email
        }
        region_codes = _load_region_codes(conn)
        addr_index = _load_addresses(conn, region_codes)

    customers: list[Customer] = []
    for r in cust_rows:
        eav = cust_eav.get(r.entity_id, {})
        firstname = eav.get("firstname", "") or ""
        lastname = eav.get("lastname", "") or ""

        addr_id = eav.get("default_billing")
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


def _load_customer_eav(conn) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for tbl, mapping in (
        ("customer_entity_varchar", CUST_ATTR_VARCHAR),
        ("customer_entity_int",     CUST_ATTR_INT),
    ):
        ids = ",".join(str(i) for i in mapping)
        for r in conn.execute(text(
            f"SELECT entity_id, attribute_id, value FROM {tbl} "
            f"WHERE attribute_id IN ({ids})"
        )):
            out.setdefault(r.entity_id, {})[mapping[r.attribute_id]] = r.value
    return out


def _load_region_codes(conn) -> dict[int, str]:
    # See magento1/orders.py for rationale — M1's region.code matches OC zone codes 1:1.
    out: dict[int, str] = {}
    for r in conn.execute(text(
        "SELECT region_id, code FROM directory_country_region WHERE code IS NOT NULL AND code <> ''"
    )):
        out[r.region_id] = r.code
    return out


def _load_addresses(conn, region_codes: dict[int, str]) -> dict[int, Address]:
    parent_index: dict[int, int] = {}
    for r in conn.execute(text(
        "SELECT entity_id, parent_id FROM customer_address_entity"
    )):
        parent_index[r.entity_id] = r.parent_id

    eav: dict[int, dict[str, str]] = {}
    for tbl, mapping in (
        ("customer_address_entity_varchar", ADDR_ATTR_VARCHAR),
        ("customer_address_entity_text",    ADDR_ATTR_TEXT),
        ("customer_address_entity_int",     ADDR_ATTR_INT),
    ):
        ids = ",".join(str(i) for i in mapping)
        for r in conn.execute(text(
            f"SELECT entity_id, attribute_id, value FROM {tbl} "
            f"WHERE attribute_id IN ({ids})"
        )):
            eav.setdefault(r.entity_id, {})[mapping[r.attribute_id]] = r.value

    out: dict[int, Address] = {}
    for addr_id, parent_id in parent_index.items():
        v = eav.get(addr_id, {})
        street = v.get("street", "") or ""
        lines = street.split("\n")
        address_1 = lines[0] if lines else ""
        address_2 = " ".join(lines[1:]) if len(lines) > 1 else ""
        try:
            region_id = int(v.get("region_id") or 0)
        except (TypeError, ValueError):
            region_id = 0
        out[addr_id] = Address(
            source_id=addr_id,
            customer_source_id=parent_id,
            firstname=v.get("firstname", "") or "",
            lastname=v.get("lastname", "") or "",
            company=v.get("company", "") or "",
            address_1=address_1,
            address_2=address_2,
            city=v.get("city", "") or "",
            postcode=v.get("postcode", "") or "",
            country_iso=(v.get("country_id") or "").upper(),
            zone_code=region_codes.get(region_id, ""),
            telephone=v.get("telephone", "") or "",
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
