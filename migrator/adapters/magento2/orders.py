"""Read Magento 2.x orders into canonical Order objects.

M2 dropped the `sales_flat_*` naming convention from M1 — tables are
`sales_order`, `sales_order_address`, `sales_order_item`, `sales_order_payment`.
Column layout is broadly the same.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Order, OrderAddressSnapshot, OrderItem


def load_orders(engine: Engine, prefix: str) -> list[Order]:
    """prefix unused for M2 (no table prefix)."""
    with engine.connect() as conn:
        order_rows = conn.execute(text(
            "SELECT entity_id, increment_id, state, status, store_id, "
            "       customer_id, customer_is_guest, customer_email, "
            "       customer_firstname, customer_lastname, customer_note, "
            "       base_subtotal, subtotal, "
            "       base_shipping_amount, shipping_amount, "
            "       base_tax_amount, tax_amount, "
            "       base_grand_total, grand_total, "
            "       order_currency_code, store_currency_code, "
            "       base_to_order_rate, "
            "       shipping_method, "
            "       created_at, updated_at "
            "FROM sales_order "
            "ORDER BY entity_id"
        )).fetchall()

        addr_index = _load_addresses(conn)
        items_index = _load_items(conn)
        payments = _load_payments(conn)

    orders: list[Order] = []
    for r in order_rows:
        addrs = addr_index.get(r.entity_id, {})
        billing = addrs.get("billing")
        shipping = addrs.get("shipping") or billing

        pay_code, pay_name = payments.get(r.entity_id, ("", ""))

        currency = (r.order_currency_code or r.store_currency_code or "EUR").upper()
        currency_value = float(r.base_to_order_rate) if r.base_to_order_rate else 1.0

        orders.append(Order(
            source_id=r.entity_id,
            reference=(r.increment_id or str(r.entity_id))[:26],
            customer_source_id=(None if r.customer_is_guest else r.customer_id),
            customer_email=(r.customer_email or "").strip(),
            customer_firstname=(r.customer_firstname or "").strip(),
            customer_lastname=(r.customer_lastname or "").strip(),
            payment_method_name=pay_name,
            payment_method_code=pay_code,
            currency_code=currency,
            currency_value=currency_value,
            subtotal=float(r.subtotal or 0.0),
            shipping_amount=float(r.shipping_amount or 0.0),
            tax_amount=float(r.tax_amount or 0.0),
            total=float(r.grand_total or 0.0),
            status_source=(r.status or ""),
            payment_address=billing,
            shipping_address=shipping,
            items=items_index.get(r.entity_id, []),
            created_at=_parse_dt(r.created_at),
            modified_at=_parse_dt(r.updated_at),
        ))

    return orders


def _load_addresses(conn) -> dict[int, dict[str, OrderAddressSnapshot]]:
    out: dict[int, dict[str, OrderAddressSnapshot]] = {}
    for r in conn.execute(text(
        "SELECT parent_id, address_type, firstname, lastname, company, "
        "       street, city, postcode, country_id, region, telephone "
        "FROM sales_order_address"
    )):
        if r.address_type not in ("billing", "shipping"):
            continue
        street = r.street or ""
        lines = street.split("\n")
        snap = OrderAddressSnapshot(
            firstname=(r.firstname or ""),
            lastname=(r.lastname or ""),
            company=(r.company or ""),
            address_1=(lines[0] if lines else ""),
            address_2=(" ".join(lines[1:]) if len(lines) > 1 else ""),
            city=(r.city or ""),
            postcode=(r.postcode or ""),
            country_iso=(r.country_id or "").upper(),
            country_name="",
            zone_name=(r.region or ""),
            telephone=(r.telephone or ""),
        )
        out.setdefault(r.parent_id, {})[r.address_type] = snap
    return out


def _load_items(conn) -> dict[int, list[OrderItem]]:
    out: dict[int, list[OrderItem]] = {}
    rows = conn.execute(text(
        "SELECT order_id, product_id, product_type, sku, name, "
        "       qty_ordered, price, tax_percent, row_total "
        "FROM sales_order_item "
        "WHERE (product_type IS NULL OR product_type IN ('simple','virtual','downloadable')) "
        "ORDER BY item_id"
    )).fetchall()
    for r in rows:
        out.setdefault(r.order_id, []).append(OrderItem(
            product_source_id=r.product_id,
            product_reference=(r.sku or "")[:64],
            name=(r.name or "")[:255],
            quantity=int(float(r.qty_ordered or 0)),
            unit_price=float(r.price or 0.0),
            tax_rate=float(r.tax_percent or 0.0),
            line_total=float(r.row_total or 0.0),
        ))
    return out


_PAYMENT_LABELS = {
    "cashondelivery":  "Αντικαταβολή",
    "paypal_express":  "PayPal",
    "paypal_standard": "PayPal",
    "checkmo":         "Επιταγή",
    "banktransfer":    "Τραπεζική Μεταφορά",
    "free":            "Δωρεάν",
    "stripe_payments": "Πιστωτική κάρτα",
    "braintree":       "Πιστωτική κάρτα",
}


def _load_payments(conn) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for r in conn.execute(text(
        "SELECT parent_id, method FROM sales_order_payment"
    )):
        code = r.method or ""
        label = _PAYMENT_LABELS.get(code, code.replace("_", " ").title())
        out[r.parent_id] = (code, label)
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
