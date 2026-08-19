"""Write canonical Order objects into OpenCart 3.x."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Order, OrderAddressSnapshot
from migrator.pipeline.id_map import (
    IdMapStore, ENTITY_ORDER, chunked, DEFAULT_BATCH_SIZE,
)

DEFAULT_LANGUAGE_ID = 1
DEFAULT_CUSTOMER_GROUP_ID = 1

# Source status code → OC order_status_id. Keyed by string for cross-platform support.

# PS current_state → OC order_status_id. Store-specific to eaccess_pres668
# (PrestaShop state IDs and labels are shop-configurable).
PS_STATUS_MAP: dict[str, int] = {
    "2":  2,   # Πληρωμή ολοκληρώθηκε (Payment complete) → Processing
    "3":  2,   # Σε επεξεργασία (In processing) → Processing
    "4":  5,   # Παραδόθηκε (Delivered) → Complete
    "5":  3,   # Έχει αποσταλεί (Has shipped) → Shipped
    "6":  7,   # Ακυρώθηκε (Canceled) → Canceled
    "7":  11,  # Επιστροφή αξίας (Refund) → Refunded
    "8":  10,  # Σφάλμα Πληρωμής (Payment error) → Failed
    "9":  2,   # Σε προ-παραγγελία πληρωμένο (Backorder paid) → Processing
    "10": 1,   # Αναμονή Κατάθεσης (Awaiting deposit) → Pending
    "12": 1,   # Εκτός αποθέματος (Backorder/Out of stock) → Pending
    "13": 1,   # Σε αναμονή (Pending) → Pending
    "20": 5,   # Πληρωμή στο κατάστημα (Pay in store) → Complete
}

# WooCommerce order status (wp_posts.post_status with 'wc-' prefix stripped) → OC order_status_id
WC_STATUS_MAP: dict[str, int] = {
    "pending":    1,  # Pending
    "on-hold":    1,  # Pending (awaiting payment confirmation)
    "processing": 2,  # Processing
    "completed":  5,  # Complete
    "cancelled":  7,  # Canceled
    "refunded":   11, # Refunded
    "failed":     10, # Failed
}

# Magento 2.x order status → OC order_status_id. The set of status codes is the
# same as M1 in practice (mostly canonical Magento codes), plus a few M2-specific
# additions like pending_paypal and the unified PayPal review codes.
M2_STATUS_MAP: dict[str, int] = {
    "pending":                   1,  # Pending
    "pending_payment":           1,  # Pending
    "pending_paypal":            1,  # Pending
    "payment_review":            1,  # Pending
    "holded":                    1,  # Pending
    "processing":                2,  # Processing
    "complete":                  5,  # Complete
    "closed":                    5,  # Complete (refunded but final)
    "canceled":                  7,  # Canceled
    "fraud":                     8,  # Denied
    "paypal_canceled_reversal":  2,  # Processing
    "paypal_reversed":           11, # Refunded
}

# Magento 1.x order status → OC order_status_id
M1_STATUS_MAP: dict[str, int] = {
    "pending":                   1,  # Pending
    "pending_payment":           1,  # Pending
    "processing":                2,  # Processing
    "canceled":                  7,  # Canceled
    "paypal_canceled_reversal":  2,  # Processing
    "complete":                  5,  # Complete
    "closed":                    5,  # Complete (refunded but final)
    "fraud":                     8,  # Denied
    "holded":                    1,  # Pending
    "payment_review":            1,  # Pending
}


def write_orders(
    engine: Engine,
    orders: list[Order],
    prefix: str,
    customer_id_map: dict[int, int],
    product_id_map: dict[int, int],
    status_map: dict[str, int],
    store_id: int = 0,
    language_id: int = DEFAULT_LANGUAGE_ID,
    customer_group_id: int = DEFAULT_CUSTOMER_GROUP_ID,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_every: int = 100,
    id_map_store: Optional[IdMapStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, int]:
    """Insert orders + products + totals + history. Returns {source_id: oc_order_id}.

    Idempotent: skips orders already in `id_map_store`. Commits per batch so a
    crash mid-run only loses up to `batch_size` orders.
    """
    existing = id_map_store.load(ENTITY_ORDER) if id_map_store else {}
    id_map: dict[int, int] = dict(existing)
    skipped_empty = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Reference lookups fetched once outside the batch transactions so each
    # batch only does write work.
    with engine.connect() as conn:
        currency_lookup = {
            r.code.upper(): (r.currency_id, r.symbol_left or "", r.symbol_right or "")
            for r in conn.execute(text(
                f"SELECT currency_id, code, symbol_left, symbol_right FROM `{prefix}currency`"
            ))
        }
        country_lookup = {
            r.iso_code_2: (r.country_id, r.name)
            for r in conn.execute(text(
                f"SELECT country_id, iso_code_2, name FROM `{prefix}country` WHERE status = 1"
            ))
        }
        zone_lookup = {
            (r.country_id, r.code.upper()): r.zone_id
            for r in conn.execute(text(
                f"SELECT zone_id, country_id, code FROM `{prefix}zone` WHERE status = 1 AND code <> ''"
            ))
        }
        store_info = conn.execute(text(
            f"SELECT name, url FROM `{prefix}store` WHERE store_id = :sid"
        ), {"sid": store_id}).first()
        store_name = store_info.name if store_info else "Default"
        store_url = store_info.url if store_info else ""

    todo = [o for o in orders if o.source_id not in existing]
    if log_cb and existing:
        log_cb(f"  • Already migrated: {len(existing)} orders — skipping")
    if not todo:
        if progress_cb:
            progress_cb(len(existing), len(existing) or 1)
        return id_map

    total = len(todo)
    processed = 0

    for chunk in chunked(todo, batch_size):
        with engine.begin() as conn:
            for o in chunk:
                if not o.items:
                    skipped_empty += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue

                oc_customer_id = customer_id_map.get(o.customer_source_id or -1, 0)
                oc_status_id = status_map.get(o.status_source, 0)
                currency_id, _sym_left, _sym_right = currency_lookup.get(o.currency_code, (0, "", ""))
                created = o.created_at.strftime("%Y-%m-%d %H:%M:%S") if o.created_at else now
                modified = o.modified_at.strftime("%Y-%m-%d %H:%M:%S") if o.modified_at else created

                ship = o.shipping_address or OrderAddressSnapshot()
                pay = o.payment_address or ship

                p_country_id, p_country_oc_name = country_lookup.get(pay.country_iso, (0, "")) if pay.country_iso else (0, "")
                p_zone_id = zone_lookup.get((p_country_id, (pay.zone_code or "").upper()), 0) if pay.zone_code else 0
                s_country_id, s_country_oc_name = country_lookup.get(ship.country_iso, (0, "")) if ship.country_iso else (0, "")
                s_zone_id = zone_lookup.get((s_country_id, (ship.zone_code or "").upper()), 0) if ship.zone_code else 0

                result = conn.execute(text(
                    f"INSERT INTO `{prefix}order` ("
                    f"  invoice_no, invoice_prefix, store_id, store_name, store_url, "
                    f"  customer_id, customer_group_id, firstname, lastname, email, telephone, fax, custom_field, "
                    f"  payment_firstname, payment_lastname, payment_company, payment_address_1, payment_address_2, "
                    f"  payment_city, payment_postcode, payment_country, payment_country_id, payment_zone, payment_zone_id, "
                    f"  payment_address_format, payment_custom_field, payment_method, payment_code, "
                    f"  shipping_firstname, shipping_lastname, shipping_company, shipping_address_1, shipping_address_2, "
                    f"  shipping_city, shipping_postcode, shipping_country, shipping_country_id, shipping_zone, shipping_zone_id, "
                    f"  shipping_address_format, shipping_custom_field, shipping_method, shipping_code, "
                    f"  comment, total, order_status_id, affiliate_id, commission, marketing_id, tracking, language_id, "
                    f"  currency_id, currency_code, currency_value, ip, forwarded_ip, user_agent, accept_language, "
                    f"  date_added, date_modified"
                    f") VALUES ("
                    f"  0, :ref, :store_id, :store_name, :store_url, "
                    f"  :cust_id, :grp, :fn, :ln, :email, :tel, '', '', "
                    f"  :p_fn, :p_ln, :p_co, :p_a1, :p_a2, "
                    f"  :p_city, :p_pc, :p_country, :p_cn_id, :p_zone, :p_zn_id, "
                    f"  '', '', :pay_name, :pay_code, "
                    f"  :s_fn, :s_ln, :s_co, :s_a1, :s_a2, "
                    f"  :s_city, :s_pc, :s_country, :s_cn_id, :s_zone, :s_zn_id, "
                    f"  '', '', '', '', "
                    f"  '', :total, :status_id, 0, 0, 0, '', :lang, "
                    f"  :cur_id, :cur_code, :cur_val, '', '', '', '', "
                    f"  :created, :modified)"
                ), {
                    "ref": _trunc(o.reference, 26),
                    "store_id": store_id, "store_name": _trunc(store_name, 64),
                    "store_url": _trunc(store_url, 255),
                    "cust_id": oc_customer_id, "grp": customer_group_id,
                    "fn": _trunc(o.customer_firstname, 32),
                    "ln": _trunc(o.customer_lastname, 32),
                    "email": _trunc(o.customer_email, 96),
                    "tel": _trunc(pay.telephone or ship.telephone, 32),
                    "p_fn": _trunc(pay.firstname, 32), "p_ln": _trunc(pay.lastname, 32),
                    "p_co": _trunc(pay.company, 60),
                    "p_a1": _trunc(pay.address_1, 128), "p_a2": _trunc(pay.address_2, 128),
                    "p_city": _trunc(pay.city, 128), "p_pc": _trunc(pay.postcode, 10),
                    "p_country": _trunc(pay.country_name or p_country_oc_name, 128),
                    "p_cn_id": p_country_id,
                    "p_zone": _trunc(pay.zone_name, 128),
                    "p_zn_id": p_zone_id,
                    "pay_name": _trunc(o.payment_method_name, 128),
                    "pay_code": _trunc(o.payment_method_code, 128),
                    "s_fn": _trunc(ship.firstname, 32), "s_ln": _trunc(ship.lastname, 32),
                    "s_co": _trunc(ship.company, 40),
                    "s_a1": _trunc(ship.address_1, 128), "s_a2": _trunc(ship.address_2, 128),
                    "s_city": _trunc(ship.city, 128), "s_pc": _trunc(ship.postcode, 10),
                    "s_country": _trunc(ship.country_name or s_country_oc_name, 128),
                    "s_cn_id": s_country_id,
                    "s_zone": _trunc(ship.zone_name, 128),
                    "s_zn_id": s_zone_id,
                    "total": round(o.total, 4), "status_id": oc_status_id, "lang": language_id,
                    "cur_id": currency_id, "cur_code": o.currency_code,
                    "cur_val": round(o.currency_value, 8),
                    "created": created, "modified": modified,
                })
                oc_order_id = result.lastrowid
                id_map[o.source_id] = oc_order_id

                for item in o.items:
                    oc_product_id = product_id_map.get(item.product_source_id or -1, 0)
                    conn.execute(text(
                        f"INSERT INTO `{prefix}order_product` "
                        f"(order_id, product_id, name, model, quantity, price, total, tax, reward) "
                        f"VALUES (:oid, :pid, :name, :model, :qty, :price, :total, :tax, 0)"
                    ), {
                        "oid": oc_order_id, "pid": oc_product_id,
                        "name": _trunc(item.name, 255),
                        "model": _trunc(item.product_reference or "", 64),
                        "qty": item.quantity,
                        "price": round(item.unit_price, 4),
                        "total": round(item.line_total, 4),
                        "tax": round(item.unit_price * item.tax_rate / 100.0, 4),
                    })

                _insert_total(conn, prefix, oc_order_id, "sub_total", "Sub-Total", o.subtotal, 1)
                if o.shipping_amount:
                    _insert_total(conn, prefix, oc_order_id, "shipping", "Shipping", o.shipping_amount, 3)
                if o.tax_amount:
                    _insert_total(conn, prefix, oc_order_id, "tax", "Tax", o.tax_amount, 5)
                _insert_total(conn, prefix, oc_order_id, "total", "Total", o.total, 9)

                conn.execute(text(
                    f"INSERT INTO `{prefix}order_history` "
                    f"(order_id, order_status_id, notify, comment, date_added) "
                    f"VALUES (:oid, :sid, 0, '', :da)"
                ), {"oid": oc_order_id, "sid": oc_status_id, "da": modified})

                if id_map_store:
                    id_map_store.record(conn, ENTITY_ORDER, o.source_id, oc_order_id)

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % log_every == 0 or processed == total):
                    log_cb(f"  ✓ [{processed}/{total}] last: {o.reference}")

    if log_cb:
        if skipped_empty:
            log_cb(f"  ⚠ Skipped {skipped_empty} orders with no line items")
        log_cb(f"  ⚠ Orders without a matching OC customer were saved with customer_id=0 (snapshot data preserved)")

    return id_map


def _insert_total(conn, prefix: str, order_id: int, code: str, title: str, value: float, sort: int) -> None:
    conn.execute(text(
        f"INSERT INTO `{prefix}order_total` "
        f"(order_id, code, title, value, sort_order) "
        f"VALUES (:oid, :code, :title, :value, :sort)"
    ), {"oid": order_id, "code": code, "title": title, "value": round(value, 4), "sort": sort})


def _trunc(s: Optional[str], max_chars: int) -> str:
    if not s:
        return ""
    return s[:max_chars] if len(s) > max_chars else s
