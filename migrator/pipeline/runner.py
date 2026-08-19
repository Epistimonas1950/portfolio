"""Orchestrates a full migration run for one source → OpenCart 3.x."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy.engine import Engine

from migrator.canonical.models import SourcePlatform
from migrator.pipeline.id_map import IdMapStore


_ADAPTER_PACKAGES = {
    SourcePlatform.PRESTASHOP:  "migrator.adapters.prestashop",
    SourcePlatform.MAGENTO1:    "migrator.adapters.magento1",
    SourcePlatform.MAGENTO2:    "migrator.adapters.magento2",
    SourcePlatform.WOOCOMMERCE: "migrator.adapters.woocommerce",
}

# Platforms that do not have a manufacturer concept (or this dump doesn't use one).
_NO_MANUFACTURERS = {
    SourcePlatform.MAGENTO1,
    SourcePlatform.MAGENTO2,
    SourcePlatform.WOOCOMMERCE,
}


@dataclass
class SourceConfig:
    platform: SourcePlatform
    engine: Engine
    table_prefix: str
    lang_map: dict[int, int]
    image_root: Optional[Path] = None


@dataclass
class TargetConfig:
    engine: Engine
    table_prefix: str
    oc_image_dir: Optional[Path] = None
    store_id: int = 0
    tax_class_id: int = 9


@dataclass
class MigrationOptions:
    categories: bool = True
    manufacturers: bool = False
    products: bool = False
    customers: bool = False
    orders: bool = False


class MigrationRunner:
    def __init__(
        self,
        source: SourceConfig,
        target: TargetConfig,
        options: MigrationOptions,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.source = source
        self.target = target
        self.options = options
        self.log = log_cb or (lambda msg: None)
        self.progress = progress_cb or (lambda label, cur, total: None)

    def run(self) -> dict:
        stats: dict[str, int] = {}
        category_id_map: dict[int, int] = {}
        manufacturer_id_map: dict[int, int] = {}
        product_id_map: dict[int, int] = {}
        customer_id_map: dict[int, int] = {}

        # Sidecar id_map enables resume-safe re-runs: each writer records
        # source_id → oc_id mappings inside its own transaction so a second run
        # skips rows it already inserted.
        self.id_map_store = IdMapStore(
            self.target.engine,
            self.target.table_prefix,
            self.source.platform.value,
        )
        self.id_map_store.ensure_table()

        run_categories = self.options.categories or self.options.products or self.options.orders
        run_manufacturers = self.options.manufacturers or self.options.products or self.options.orders
        run_products = self.options.products or self.options.orders
        run_customers = self.options.customers or self.options.orders

        if run_categories:
            category_id_map = self._migrate_categories()
            stats["categories"] = len(category_id_map)

        if run_manufacturers:
            manufacturer_id_map = self._migrate_manufacturers()
            stats["manufacturers"] = len(manufacturer_id_map)

        if run_products:
            product_id_map = self._migrate_products(category_id_map, manufacturer_id_map)
            stats["products"] = len(product_id_map)

        if run_customers:
            customer_id_map = self._migrate_customers()
            stats["customers"] = len(customer_id_map)

        if self.options.orders:
            count = self._migrate_orders(customer_id_map, product_id_map)
            stats["orders"] = count

        return stats

    # ── platform dispatch ─────────────────────────────────────

    def _adapter(self, name: str):
        pkg = _ADAPTER_PACKAGES.get(self.source.platform)
        if not pkg:
            self.log(f"  ✗ No adapter package for {self.source.platform}")
            return None
        try:
            return import_module(f"{pkg}.{name}")
        except ModuleNotFoundError:
            self.log(f"  ✗ {name} adapter not yet implemented for {self.source.platform}")
            return None

    def _status_map(self) -> dict[str, int]:
        from migrator.writers.opencart3.orders import (
            PS_STATUS_MAP, M1_STATUS_MAP, M2_STATUS_MAP, WC_STATUS_MAP,
        )
        if self.source.platform == SourcePlatform.MAGENTO1:
            return M1_STATUS_MAP
        if self.source.platform == SourcePlatform.MAGENTO2:
            return M2_STATUS_MAP
        if self.source.platform == SourcePlatform.WOOCOMMERCE:
            return WC_STATUS_MAP
        return PS_STATUS_MAP

    # ── entity migrations ─────────────────────────────────────

    def _migrate_categories(self) -> dict[int, int]:
        self.log("── Categories ─────────────────────────")
        src, tgt = self.source, self.target
        mod = self._adapter("categories")
        if mod is None:
            return {}
        from migrator.writers.opencart3.categories import write_categories

        self.log("  Loading categories from source…")
        cats = mod.load_categories(src.engine, src.table_prefix, src.lang_map, src.image_root)
        self.log(f"  Found {len(cats)} categories")
        self.progress("Categories", 0, len(cats))

        return write_categories(
            tgt.engine, cats, tgt.table_prefix,
            tgt.oc_image_dir, tgt.store_id,
            log_cb=self.log,
            progress_cb=lambda c, t: self.progress("Categories", c, t),
            id_map_store=self.id_map_store,
        )

    def _migrate_manufacturers(self) -> dict[int, int]:
        self.log("── Manufacturers ──────────────────────")
        src, tgt = self.source, self.target

        if src.platform in _NO_MANUFACTURERS:
            self.log(f"  • {src.platform.value} has no manufacturer concept — skipping")
            return {}

        mod = self._adapter("manufacturers")
        if mod is None:
            return {}
        from migrator.writers.opencart3.manufacturers import write_manufacturers

        self.log("  Loading manufacturers from source…")
        manus = mod.load_manufacturers(src.engine, src.table_prefix, src.image_root)
        self.log(f"  Found {len(manus)} manufacturers")
        self.progress("Manufacturers", 0, len(manus))

        return write_manufacturers(
            tgt.engine, manus, tgt.table_prefix,
            tgt.oc_image_dir, tgt.store_id,
            log_cb=self.log,
            progress_cb=lambda c, t: self.progress("Manufacturers", c, t),
            id_map_store=self.id_map_store,
        )

    def _migrate_products(
        self,
        category_id_map: dict[int, int],
        manufacturer_id_map: dict[int, int],
    ) -> dict[int, int]:
        self.log("── Products ───────────────────────────")
        src, tgt = self.source, self.target
        mod = self._adapter("products")
        if mod is None:
            return {}
        from migrator.writers.opencart3.products import write_products

        self.log("  Loading products from source (this may take a moment)…")
        products = mod.load_products(src.engine, src.table_prefix, src.lang_map, src.image_root)
        self.log(f"  Found {len(products)} products")
        self.progress("Products", 0, len(products))

        id_map = write_products(
            tgt.engine, products, tgt.table_prefix,
            category_id_map=category_id_map,
            manufacturer_id_map=manufacturer_id_map,
            oc_image_dir=tgt.oc_image_dir,
            store_id=tgt.store_id,
            tax_class_id=tgt.tax_class_id,
            log_cb=self.log,
            progress_cb=lambda c, t: self.progress("Products", c, t),
            id_map_store=self.id_map_store,
        )
        self.log(f"  Done — {len(id_map)} products written")
        return id_map

    def _migrate_customers(self) -> dict[int, int]:
        self.log("── Customers ──────────────────────────")
        src, tgt = self.source, self.target
        mod = self._adapter("customers")
        if mod is None:
            return {}
        from migrator.writers.opencart3.customers import write_customers

        self.log("  Loading customers from source…")
        customers = mod.load_customers(src.engine, src.table_prefix)
        self.log(f"  Found {len(customers)} customers")
        self.progress("Customers", 0, len(customers))

        id_map = write_customers(
            tgt.engine, customers, tgt.table_prefix,
            store_id=tgt.store_id,
            log_cb=self.log,
            progress_cb=lambda c, t: self.progress("Customers", c, t),
            id_map_store=self.id_map_store,
        )
        self.log(f"  Done — {len(id_map)} customers written")
        return id_map

    def _migrate_orders(
        self,
        customer_id_map: dict[int, int],
        product_id_map: dict[int, int],
    ) -> int:
        self.log("── Orders ─────────────────────────────")
        src, tgt = self.source, self.target
        mod = self._adapter("orders")
        if mod is None:
            return 0
        from migrator.writers.opencart3.orders import write_orders

        self.log("  Loading orders from source (this may take a moment)…")
        orders = mod.load_orders(src.engine, src.table_prefix)
        self.log(f"  Found {len(orders)} orders")
        self.progress("Orders", 0, len(orders))

        id_map = write_orders(
            tgt.engine, orders, tgt.table_prefix,
            customer_id_map=customer_id_map,
            product_id_map=product_id_map,
            status_map=self._status_map(),
            store_id=tgt.store_id,
            log_cb=self.log,
            progress_cb=lambda c, t: self.progress("Orders", c, t),
            id_map_store=self.id_map_store,
        )
        self.log(f"  Done — {len(id_map)} orders written")
        return len(id_map)
