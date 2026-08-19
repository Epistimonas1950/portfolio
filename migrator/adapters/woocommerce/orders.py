"""Read WooCommerce orders into canonical Order objects."""
from __future__ import annotations
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Order, OrderAddressSnapshot, OrderItem

ORDER_META_KEYS = (
    "_customer_user", "_order_currency",
    "_order_total", "_order_tax", "_order_shipping", "_order_shipping_tax",
    "_cart_discount", "_cart_discount_tax",
    "_payment_method", "_payment_method_title",
    "_billing_first_name", "_billing_last_name", "_billing_company",
    "_billing_address_1", "_billing_address_2",
    "_billing_city", "_billing_state", "_billing_postcode", "_billing_country",
    "_billing_email", "_billing_phone",
    "_shipping_first_name", "_shipping_last_name", "_shipping_company",
    "_shipping_address_1", "_shipping_address_2",
    "_shipping_city", "_shipping_state", "_shipping_postcode", "_shipping_country",
)

ITEM_META_KEYS = ("_product_id", "_qty", "_line_total", "_line_subtotal", "_line_tax")


def load_orders(engine: Engine, prefix: str) -> list[Order]:
    with engine.connect() as conn:
        order_rows = conn.execute(text(
            f"SELECT ID, post_status, post_date, post_modified "
            f"FROM `{prefix}posts` "
            f"WHERE post_type = 'shop_order' "
            f"ORDER BY ID"
        )).fetchall()
        order_ids = [r.ID for r in order_rows]
        if not order_ids:
            return []

        meta = _load_order_meta(conn, prefix, order_ids)
        items_index = _load_items(conn, prefix, order_ids)

    orders: list[Order] = []
    for r in order_rows:
        m = meta.get(r.ID, {})

        billing = _build_address(m, "_billing_")
        shipping = _build_address(m, "_shipping_") or billing

        status = (r.post_status or "").removeprefix("wc-")

        customer_id_raw = m.get("_customer_user") or "0"
        customer_id = int(customer_id_raw) if customer_id_raw.isdigit() and int(customer_id_raw) > 0 else None

        total = _to_float(m.get("_order_total") or 0)
        shipping_amt = _to_float(m.get("_order_shipping") or 0)
        tax = _to_float(m.get("_order_tax") or 0) + _to_float(m.get("_order_shipping_tax") or 0)
        items = items_index.get(r.ID, [])
        subtotal = sum(it.line_total for it in items)

        orders.append(Order(
            source_id=r.ID,
            reference=str(r.ID),
            customer_source_id=customer_id,
            customer_email=(m.get("_billing_email") or "").strip(),
            customer_firstname=(m.get("_billing_first_name") or "").strip(),
            customer_lastname=(m.get("_billing_last_name") or "").strip(),
            payment_method_name=(m.get("_payment_method_title") or "").strip(),
            payment_method_code=(m.get("_payment_method") or "").strip(),
            currency_code=(m.get("_order_currency") or "EUR").upper(),
            currency_value=1.0,
            subtotal=subtotal,
            shipping_amount=shipping_amt,
            tax_amount=tax,
            total=total,
            status_source=status,
            payment_address=billing,
            shipping_address=shipping,
            items=items,
            created_at=_parse_dt(r.post_date),
            modified_at=_parse_dt(r.post_modified),
        ))

    return orders


def _build_address(m: dict[str, str], prefix: str) -> Optional[OrderAddressSnapshot]:
    addr1 = (m.get(prefix + "address_1") or "").strip()
    city = (m.get(prefix + "city") or "").strip()
    postcode = (m.get(prefix + "postcode") or "").strip()
    if not (addr1 or city or postcode):
        return None
    return OrderAddressSnapshot(
        firstname=(m.get(prefix + "first_name") or "").strip(),
        lastname=(m.get(prefix + "last_name") or "").strip(),
        company=(m.get(prefix + "company") or "").strip(),
        address_1=addr1,
        address_2=(m.get(prefix + "address_2") or "").strip(),
        city=city,
        postcode=postcode,
        country_iso=(m.get(prefix + "country") or "").strip().upper(),
        country_name="",
        zone_name=(m.get(prefix + "state") or "").strip(),
        telephone=(m.get("_billing_phone") or "").strip(),
    )


def _load_order_meta(conn, prefix: str, ids: list[int]) -> dict[int, dict[str, str]]:
    placeholders = ",".join(str(i) for i in ids)
    key_list = ",".join(f"'{k}'" for k in ORDER_META_KEYS)
    out: dict[int, dict[str, str]] = {}
    for r in conn.execute(text(
        f"SELECT post_id, meta_key, meta_value FROM `{prefix}postmeta` "
        f"WHERE post_id IN ({placeholders}) AND meta_key IN ({key_list})"
    )):
        out.setdefault(r.post_id, {})[r.meta_key] = r.meta_value
    return out


def _load_items(conn, prefix: str, order_ids: list[int]) -> dict[int, list[OrderItem]]:
    placeholders = ",".join(str(i) for i in order_ids)
    rows = conn.execute(text(
        f"SELECT order_item_id, order_id, order_item_name "
        f"FROM `{prefix}woocommerce_order_items` "
        f"WHERE order_id IN ({placeholders}) AND order_item_type = 'line_item'"
    )).fetchall()
    if not rows:
        return {}

    item_to_order: dict[int, int] = {r.order_item_id: r.order_id for r in rows}
    item_names: dict[int, str] = {r.order_item_id: (r.order_item_name or "") for r in rows}

    item_ids = ",".join(str(i) for i in item_to_order)
    key_list = ",".join(f"'{k}'" for k in ITEM_META_KEYS)
    item_meta: dict[int, dict[str, str]] = {}
    for r in conn.execute(text(
        f"SELECT order_item_id, meta_key, meta_value "
        f"FROM `{prefix}woocommerce_order_itemmeta` "
        f"WHERE order_item_id IN ({item_ids}) AND meta_key IN ({key_list})"
    )):
        item_meta.setdefault(r.order_item_id, {})[r.meta_key] = r.meta_value

    out: dict[int, list[OrderItem]] = {}
    for item_id, order_id in item_to_order.items():
        m = item_meta.get(item_id, {})
        qty = _to_int(m.get("_qty") or "1") or 1
        line_total = _to_float(m.get("_line_total") or "0")
        unit_price = line_total / qty if qty else 0.0
        product_id = _to_int(m.get("_product_id") or "0")
        out.setdefault(order_id, []).append(OrderItem(
            product_source_id=product_id if product_id > 0 else None,
            product_reference=str(product_id) if product_id > 0 else "",
            name=item_names[item_id][:255],
            quantity=qty,
            unit_price=unit_price,
            tax_rate=0.0,
            line_total=line_total,
        ))
    return out


def _to_float(v) -> float:
    try:
        return float(v) if v not in (None, "", "NULL") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _to_int(v) -> int:
    try:
        return int(float(v)) if v not in (None, "", "NULL") else 0
    except (TypeError, ValueError):
        return 0


def _parse_dt(v) -> Optional[datetime]:
    if not v or str(v).startswith("0000"):
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except Exception:
        return None
