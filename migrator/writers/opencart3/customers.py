"""Write canonical Customer objects (with addresses) into OpenCart 3.x."""
from __future__ import annotations
import secrets
import string
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Customer
from migrator.pipeline.id_map import (
    IdMapStore, ENTITY_CUSTOMER, chunked, DEFAULT_BATCH_SIZE,
)

DEFAULT_CUSTOMER_GROUP_ID = 1
DEFAULT_LANGUAGE_ID = 1
DEFAULT_STORE_ID = 0


def write_customers(
    engine: Engine,
    customers: list[Customer],
    prefix: str,
    customer_group_id: int = DEFAULT_CUSTOMER_GROUP_ID,
    language_id: int = DEFAULT_LANGUAGE_ID,
    store_id: int = DEFAULT_STORE_ID,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_every: int = 100,
    id_map_store: Optional[IdMapStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, int]:
    """Insert customers + their primary addresses. Returns {source_id: oc_customer_id}.

    Idempotent: skips customers already in `id_map_store`. Commits per batch.
    """
    existing = id_map_store.load(ENTITY_CUSTOMER) if id_map_store else {}
    id_map: dict[int, int] = dict(existing)
    skipped_no_email = 0
    skipped_duplicate = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Country + zone lookups + already-known emails fetched once outside the
    # batch transactions so each batch only does write work.
    with engine.connect() as conn:
        country_lookup = {
            r.iso_code_2: r.country_id
            for r in conn.execute(text(
                f"SELECT country_id, iso_code_2 FROM `{prefix}country` WHERE status = 1"
            ))
        }
        zone_lookup = {
            (r.country_id, r.code.upper()): r.zone_id
            for r in conn.execute(text(
                f"SELECT zone_id, country_id, code FROM `{prefix}zone` WHERE status = 1 AND code <> ''"
            ))
        }
        existing_emails = {
            r.email for r in conn.execute(text(
                f"SELECT email FROM `{prefix}customer`"
            ))
        }

    todo = [c for c in customers if c.source_id not in existing]
    if log_cb and existing:
        log_cb(f"  • Already migrated: {len(existing)} customers — skipping")
    if not todo:
        if progress_cb:
            progress_cb(len(existing), len(existing) or 1)
        return id_map

    total = len(todo)
    processed = 0

    for chunk in chunked(todo, batch_size):
        with engine.begin() as conn:
            for c in chunk:
                if not c.email:
                    skipped_no_email += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue
                if c.email in existing_emails:
                    skipped_duplicate += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue
                existing_emails.add(c.email)

                password, salt = _random_password()
                created = c.created_at.strftime("%Y-%m-%d %H:%M:%S") if c.created_at else now

                result = conn.execute(text(
                    f"INSERT INTO `{prefix}customer` "
                    f"(customer_group_id, store_id, language_id, firstname, lastname, "
                    f"email, telephone, fax, password, salt, cart, wishlist, "
                    f"newsletter, address_id, custom_field, ip, status, safe, token, code, date_added) "
                    f"VALUES (:grp, :store, :lang, :fn, :ln, "
                    f":email, :tel, '', :pwd, :salt, '', '', "
                    f":news, 0, '', '', 1, 1, '', '', :now)"
                ), {
                    "grp": customer_group_id, "store": store_id, "lang": language_id,
                    "fn": _trunc(c.firstname, 32), "ln": _trunc(c.lastname, 32),
                    "email": _trunc(c.email, 96), "tel": _trunc(c.telephone, 32),
                    "pwd": password, "salt": salt,
                    "news": 1 if c.newsletter else 0,
                    "now": created,
                })
                oc_cust_id = result.lastrowid
                id_map[c.source_id] = oc_cust_id

                if c.address:
                    a = c.address
                    country_id = country_lookup.get(a.country_iso, 0)
                    zone_id = zone_lookup.get((country_id, (a.zone_code or "").upper()), 0) if a.zone_code else 0
                    addr_result = conn.execute(text(
                        f"INSERT INTO `{prefix}address` "
                        f"(customer_id, firstname, lastname, company, "
                        f"address_1, address_2, city, postcode, country_id, zone_id, custom_field) "
                        f"VALUES (:cid, :fn, :ln, :co, :a1, :a2, :city, :pc, :cn, :zid, '')"
                    ), {
                        "cid": oc_cust_id,
                        "fn": _trunc(a.firstname or c.firstname, 32),
                        "ln": _trunc(a.lastname or c.lastname, 32),
                        "co": _trunc(a.company, 40),
                        "a1": _trunc(a.address_1, 128),
                        "a2": _trunc(a.address_2, 128),
                        "city": _trunc(a.city, 128),
                        "pc": _trunc(a.postcode, 10),
                        "cn": country_id,
                        "zid": zone_id,
                    })
                    oc_addr_id = addr_result.lastrowid

                    conn.execute(text(
                        f"UPDATE `{prefix}customer` SET address_id = :aid WHERE customer_id = :cid"
                    ), {"aid": oc_addr_id, "cid": oc_cust_id})

                if id_map_store:
                    id_map_store.record(conn, ENTITY_CUSTOMER, c.source_id, oc_cust_id)

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % log_every == 0 or processed == total):
                    log_cb(f"  ✓ [{processed}/{total}] last: {c.email}")

    if log_cb:
        if skipped_no_email:
            log_cb(f"  ⚠ Skipped {skipped_no_email} customers with no email")
        if skipped_duplicate:
            log_cb(f"  ⚠ Skipped {skipped_duplicate} customers with duplicate emails")
        log_cb("  ⚠ All migrated customers have random passwords — they must use 'Forgot Password' to log in")

    return id_map


def _random_password() -> tuple[str, str]:
    """Return (password_hash, salt). Hash is unrecoverable — customer must reset."""
    pwd = secrets.token_hex(20)
    salt = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(9))
    return pwd, salt


def _trunc(s: str, max_chars: int) -> str:
    """Truncate to fit in VARCHAR(N). Char-based — VARCHAR in utf8mb4 counts chars not bytes."""
    if not s:
        return s or ""
    return s[:max_chars] if len(s) > max_chars else s
