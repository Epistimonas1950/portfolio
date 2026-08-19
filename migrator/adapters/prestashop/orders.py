"""Read PrestaShop orders into canonical Order objects."""
from __future__ import annotations
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import (
    Order, OrderItem, OrderAddressSnapshot,
)
from migrator.adapters.prestashop._zones import ps_zone_to_oc


def load_orders(engine: Engine, prefix: str) -> list[Order]:
    with engine.connect() as conn:
        items_by_order = _load_items(conn, prefix)
        header_rows = conn.execute(text(
            f"SELECT "
            f"  o.id_order, o.reference, o.id_customer, "
            f"  o.total_paid, o.total_paid_tax_incl, o.total_paid_tax_excl, "
            f"  o.total_products, o.total_products_wt, "
            f"  o.total_shipping, o.total_shipping_tax_incl, o.total_shipping_tax_excl, "
            f"  o.current_state, o.payment, o.module, "
            f"  o.date_add, o.date_upd, "
            f"  cur.iso_code AS currency_iso, cur.conversion_rate, "
            f"  c.email AS cust_email, c.firstname AS cust_fn, c.lastname AS cust_ln, "
            f"  ad.firstname AS d_fn, ad.lastname AS d_ln, ad.company AS d_co, "
            f"  ad.address1 AS d_a1, ad.address2 AS d_a2, ad.city AS d_city, "
            f"  ad.postcode AS d_pc, ad.phone AS d_phone, ad.phone_mobile AS d_mob, "
            f"  cd.iso_code AS d_country_iso, cdl.name AS d_country_name, sd.name AS d_zone_name, sd.iso_code AS d_zone_iso, "
            f"  ai.firstname AS i_fn, ai.lastname AS i_ln, ai.company AS i_co, "
            f"  ai.address1 AS i_a1, ai.address2 AS i_a2, ai.city AS i_city, "
            f"  ai.postcode AS i_pc, ai.phone AS i_phone, ai.phone_mobile AS i_mob, "
            f"  ci.iso_code AS i_country_iso, cil.name AS i_country_name, si.name AS i_zone_name, si.iso_code AS i_zone_iso "
            f"FROM `{prefix}orders` o "
            f"LEFT JOIN `{prefix}currency` cur ON cur.id_currency = o.id_currency "
            f"LEFT JOIN `{prefix}customer` c ON c.id_customer = o.id_customer "
            f"LEFT JOIN `{prefix}address` ad ON ad.id_address = o.id_address_delivery "
            f"LEFT JOIN `{prefix}country` cd ON cd.id_country = ad.id_country "
            f"LEFT JOIN `{prefix}country_lang` cdl ON cdl.id_country = cd.id_country AND cdl.id_lang = 2 "
            f"LEFT JOIN `{prefix}state` sd ON sd.id_state = ad.id_state "
            f"LEFT JOIN `{prefix}address` ai ON ai.id_address = o.id_address_invoice "
            f"LEFT JOIN `{prefix}country` ci ON ci.id_country = ai.id_country "
            f"LEFT JOIN `{prefix}country_lang` cil ON cil.id_country = ci.id_country AND cil.id_lang = 2 "
            f"LEFT JOIN `{prefix}state` si ON si.id_state = ai.id_state "
            f"ORDER BY o.id_order"
        )).fetchall()

    out: list[Order] = []
    for r in header_rows:
        ship_addr = _snapshot(
            r.d_fn, r.d_ln, r.d_co, r.d_a1, r.d_a2, r.d_city, r.d_pc,
            r.d_country_iso, r.d_country_name, r.d_zone_name, r.d_zone_iso,
            r.d_mob or r.d_phone,
        )
        pay_addr = _snapshot(
            r.i_fn, r.i_ln, r.i_co, r.i_a1, r.i_a2, r.i_city, r.i_pc,
            r.i_country_iso, r.i_country_name, r.i_zone_name, r.i_zone_iso,
            r.i_mob or r.i_phone,
        )
        subtotal = float(r.total_products_wt or r.total_products or 0)
        shipping = float(r.total_shipping_tax_incl or r.total_shipping or 0)
        total = float(r.total_paid_tax_incl or r.total_paid or 0)
        tax = max(total - subtotal - shipping, 0.0)
        out.append(Order(
            source_id=r.id_order,
            reference=(r.reference or str(r.id_order)).strip(),
            customer_source_id=r.id_customer or None,
            customer_email=(r.cust_email or "").strip().lower(),
            customer_firstname=(r.cust_fn or "").strip(),
            customer_lastname=(r.cust_ln or "").strip(),
            payment_method_name=(r.payment or "").strip(),
            payment_method_code=(r.module or "").strip(),
            currency_code=(r.currency_iso or "EUR").upper(),
            currency_value=float(r.conversion_rate or 1.0),
            subtotal=subtotal,
            shipping_amount=shipping,
            tax_amount=tax,
            total=total,
            status_source=str(int(r.current_state or 0)),
            payment_address=pay_addr,
            shipping_address=ship_addr,
            items=items_by_order.get(r.id_order, []),
            created_at=r.date_add,
            modified_at=r.date_upd,
        ))
    return out


def _load_items(conn, prefix: str) -> dict[int, list[OrderItem]]:
    rows = conn.execute(text(
        f"SELECT id_order, product_id, product_name, product_reference, "
        f"product_quantity, unit_price_tax_excl, product_price, "
        f"tax_rate, total_price_tax_incl "
        f"FROM `{prefix}order_detail` "
        f"ORDER BY id_order, id_order_detail"
    )).fetchall()
    out: dict[int, list[OrderItem]] = defaultdict(list)
    for r in rows:
        unit = float(r.unit_price_tax_excl if r.unit_price_tax_excl is not None else (r.product_price or 0))
        qty = int(r.product_quantity or 0)
        out[r.id_order].append(OrderItem(
            product_source_id=r.product_id or None,
            product_reference=(r.product_reference or "").strip(),
            name=(r.product_name or "").strip(),
            quantity=qty,
            unit_price=unit,
            tax_rate=float(r.tax_rate or 0),
            line_total=float(r.total_price_tax_incl or (unit * qty)),
        ))
    return out


def _snapshot(fn, ln, co, a1, a2, city, pc, country_iso, country_name, zone_name, zone_iso, phone) -> OrderAddressSnapshot:
    cn_iso = ((country_iso or "")[:2]).upper()
    return OrderAddressSnapshot(
        firstname=(fn or "").strip(),
        lastname=(ln or "").strip(),
        company=(co or "").strip(),
        address_1=(a1 or "").strip(),
        address_2=(a2 or "").strip(),
        city=(city or "").strip(),
        postcode=(pc or "").strip(),
        country_iso=cn_iso,
        country_name=(country_name or "").strip(),
        zone_name=(zone_name or "").strip(),
        zone_code=ps_zone_to_oc(cn_iso, zone_iso or ""),
        telephone=(phone or "").strip(),
    )
