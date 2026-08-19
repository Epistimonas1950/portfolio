"""Write canonical Product objects into OpenCart 3.x."""
from __future__ import annotations
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Product
from migrator.pipeline.id_map import (
    IdMapStore, ENTITY_PRODUCT, chunked, DEFAULT_BATCH_SIZE,
)

DEFAULT_TAX_CLASS_ID = 9   # OC "Taxable Goods"
STOCK_IN_STOCK = 7
STOCK_OUT_OF_STOCK = 5


def write_products(
    engine: Engine,
    products: list[Product],
    prefix: str,
    category_id_map: dict[int, int],
    manufacturer_id_map: dict[int, int],
    oc_image_dir: Optional[Path] = None,
    store_id: int = 0,
    tax_class_id: int = DEFAULT_TAX_CLASS_ID,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_every: int = 25,
    id_map_store: Optional[IdMapStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, int]:
    """Insert products + descriptions + category links + images. Returns {source_id: oc_product_id}.

    Idempotent: skips products already in `id_map_store`. Commits per batch
    so a crash mid-run only loses up to `batch_size` rows.
    """
    existing = id_map_store.load(ENTITY_PRODUCT) if id_map_store else {}
    id_map: dict[int, int] = dict(existing)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    todo = [p for p in products if p.source_id not in existing]
    if log_cb and existing:
        log_cb(f"  • Already migrated: {len(existing)} products — skipping")
    if not todo:
        if progress_cb:
            progress_cb(len(existing), len(existing) or 1)
        return id_map

    total = len(todo)
    processed = 0
    skipped = 0

    for chunk in chunked(todo, batch_size):
        with engine.begin() as conn:
            for p in chunk:
                if not p.names:
                    skipped += 1
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue

                cover_val, extras = _copy_images(p, oc_image_dir)
                stock_status = STOCK_IN_STOCK if p.quantity > 0 else STOCK_OUT_OF_STOCK
                manu_id = (
                    manufacturer_id_map.get(p.manufacturer_source_id, 0)
                    if p.manufacturer_source_id else 0
                )
                date_added = (
                    p.date_added.strftime("%Y-%m-%d %H:%M:%S") if p.date_added else now
                )

                result = conn.execute(text(
                    f"INSERT INTO `{prefix}product` "
                    f"(model, sku, upc, ean, jan, isbn, mpn, location, quantity, stock_status_id, "
                    f"image, manufacturer_id, shipping, price, points, tax_class_id, date_available, "
                    f"weight, weight_class_id, length, width, height, length_class_id, "
                    f"subtract, minimum, sort_order, status, viewed, date_added, date_modified) "
                    f"VALUES (:model, :sku, :upc, :ean, '', :isbn, :mpn, '', :qty, :ss, "
                    f":img, :man, 1, :price, 0, :tax, :date_avail, "
                    f":weight, 0, :length, :width, :height, 0, "
                    f"1, :min, :sort, :status, 0, :date_added, :date_modified)"
                ), {
                    "model": p.model, "sku": p.sku, "upc": p.upc, "ean": p.ean,
                    "isbn": p.isbn, "mpn": p.mpn, "qty": p.quantity, "ss": stock_status,
                    "img": cover_val, "man": manu_id, "price": p.price, "tax": tax_class_id,
                    "date_avail": date_added[:10],
                    "weight": p.weight,
                    "length": p.depth,
                    "width": p.width, "height": p.height,
                    "min": p.minimum, "sort": p.sort_order, "status": int(p.status),
                    "date_added": date_added, "date_modified": now,
                })
                oc_id = result.lastrowid
                id_map[p.source_id] = oc_id

                for lang_id, name in p.names.items():
                    conn.execute(text(
                        f"INSERT INTO `{prefix}product_description` "
                        f"(product_id, language_id, name, description, tag, "
                        f"meta_title, meta_description, meta_keyword) "
                        f"VALUES (:pid, :lid, :name, :desc, '', :mt, :md, :mk)"
                    ), {
                        "pid": oc_id, "lid": lang_id,
                        "name": _safe_text(name, 250),
                        "desc": _safe_text(p.descriptions.get(lang_id, ""), 65000),
                        "mt": _safe_text(p.meta_titles.get(lang_id, ""), 250),
                        "md": _safe_text(p.meta_descriptions.get(lang_id, ""), 250),
                        "mk": _safe_text(p.meta_keywords.get(lang_id, ""), 250),
                    })

                conn.execute(text(
                    f"INSERT INTO `{prefix}product_to_store` (product_id, store_id) "
                    f"VALUES (:pid, :sid)"
                ), {"pid": oc_id, "sid": store_id})

                for ps_cat_id in p.category_source_ids:
                    oc_cat_id = category_id_map.get(ps_cat_id)
                    if oc_cat_id:
                        conn.execute(text(
                            f"INSERT INTO `{prefix}product_to_category` (product_id, category_id) "
                            f"VALUES (:pid, :cid)"
                        ), {"pid": oc_id, "cid": oc_cat_id})

                for sort_idx, extra_path in enumerate(extras):
                    conn.execute(text(
                        f"INSERT INTO `{prefix}product_image` (product_id, image, sort_order) "
                        f"VALUES (:pid, :img, :sort)"
                    ), {"pid": oc_id, "img": extra_path, "sort": sort_idx})

                if id_map_store:
                    id_map_store.record(conn, ENTITY_PRODUCT, p.source_id, oc_id)

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % log_every == 0 or processed == total):
                    label = next(iter(p.names.values()), p.model)
                    log_cb(f"  ✓ [{processed}/{total}] last: {label[:60]}")

    if log_cb and skipped:
        log_cb(f"  ⚠ Skipped {skipped} products with no language data in mapped languages")

    return id_map


def _safe_text(s: str, max_bytes: int) -> str:
    """Truncate string to safely fit in a TEXT/VARCHAR column at UTF-8 byte limit."""
    if not s:
        return s
    encoded = s.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _copy_images(p: Product, oc_image_dir: Optional[Path]) -> tuple[str, list[str]]:
    """Copy product images to OC. Return (cover_path, [extra_paths])."""
    if not p.images or not oc_image_dir:
        return "", []

    cover_val = ""
    extras: list[str] = []
    for img in p.images:
        if not img.source_path:
            continue
        oc_path = f"catalog/{img.dest_filename}"
        dest = oc_image_dir / oc_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(img.source_path, dest)
        except OSError:
            continue
        if img.is_cover and not cover_val:
            cover_val = oc_path
        else:
            extras.append(oc_path)
    return cover_val, extras
