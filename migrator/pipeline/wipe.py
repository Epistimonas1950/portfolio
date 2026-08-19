"""Wipe OpenCart 3.x migration tables — TRUNCATEs everything the writers populate.

Leaves config tables alone (oc_setting, oc_user, oc_language, oc_country, oc_zone,
oc_currency, oc_store, oc_store_layout, etc.). Drops demo product/category/customer
data too — that's expected for a fresh migration run.

Also TRUNCATEs the migrator_id_map sidecar so the resume-skip mappings stay
consistent with the now-empty OC tables; otherwise the next run would see source
rows as 'already migrated' and skip them.
"""
from __future__ import annotations
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

WIPE_TABLES = [
    "category", "category_description", "category_path",
    "category_to_store", "category_to_layout",
    "manufacturer", "manufacturer_to_store",
    "product", "product_description", "product_image",
    "product_to_store", "product_to_category", "product_to_layout",
    "customer", "address",
    "order", "order_product", "order_total", "order_history",
    "migrator_id_map",
]


def wipe_target(
    engine: Engine,
    prefix: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> None:
    log = log_cb or (lambda msg: None)
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for tbl in WIPE_TABLES:
            try:
                conn.execute(text(f"TRUNCATE TABLE `{prefix}{tbl}`"))
                log(f"  ✓ {prefix}{tbl}")
            except Exception as e:
                log(f"  • {prefix}{tbl}: skipped ({e.__class__.__name__})")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
