"""Read Magento 1.x products into canonical Product objects via EAV pivot."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import ImageRef, Product

# EAV attribute IDs for entity_type_id=10 (catalog_product) on this M1 install.
PROD_ATTR_VARCHAR = {96: "name", 106: "image", 109: "small_image", 493: "thumbnail",
                     103: "meta_title", 105: "meta_description", 481: "url_key"}
PROD_ATTR_TEXT    = {97: "description", 506: "short_description", 104: "meta_keyword"}
PROD_ATTR_DECIMAL = {99: "price", 101: "weight", 567: "special_price"}
PROD_ATTR_INT     = {273: "status", 526: "visibility", 274: "tax_class_id"}


def load_products(
    engine: Engine,
    prefix: str,                       # unused for M1
    lang_map: dict[int, int],          # m1_store_id → oc_lang_id
    image_root: Optional[Path] = None,
    only_active: bool = False,
) -> list[Product]:
    with engine.connect() as conn:
        prod_rows = conn.execute(text(
            "SELECT entity_id, sku, type_id, created_at "
            "FROM catalog_product_entity "
            "WHERE type_id = 'simple' "
            "ORDER BY entity_id"
        )).fetchall()

        eav = _load_eav(conn)
        stock = _load_stock(conn)
        cat_links = _load_category_links(conn)
        images = _load_images(conn)

    products: list[Product] = []
    for r in prod_rows:
        per_store = eav.get(r.entity_id, {})

        names, descriptions = {}, {}
        meta_titles, meta_descs, meta_kws = {}, {}, {}

        for m1_store, oc_lang in lang_map.items():
            name = _pick(per_store, m1_store, "name")
            if not name:
                continue
            names[oc_lang] = name
            short = _pick(per_store, m1_store, "short_description", "") or ""
            long_ = _pick(per_store, m1_store, "description", "") or ""
            descriptions[oc_lang] = (short + "\n" + long_).strip() if short and long_ else (long_ or short)
            meta_titles[oc_lang] = _pick(per_store, m1_store, "meta_title", "") or ""
            meta_descs[oc_lang] = _pick(per_store, m1_store, "meta_description", "") or ""
            meta_kws[oc_lang] = _pick(per_store, m1_store, "meta_keyword", "") or ""

        if not names:
            continue

        status_raw = _pick(per_store, 0, "status", "1")
        status_enabled = (str(status_raw) == "1")
        if only_active and not status_enabled:
            continue

        price_raw = _pick(per_store, 0, "price", "0") or "0"
        weight_raw = _pick(per_store, 0, "weight", "0") or "0"

        qty_raw = stock.get(r.entity_id, "0")

        image_refs: list[ImageRef] = []
        seen_paths: set[str] = set()
        cover_rel = _pick(per_store, 0, "image")
        if cover_rel and cover_rel != "no_selection":
            ref = _make_image_ref(r.entity_id, names, cover_rel, image_root, is_cover=True)
            if ref:
                image_refs.append(ref)
                seen_paths.add(ref.source_path)
        for gal_rel in images.get(r.entity_id, []):
            ref = _make_image_ref(r.entity_id, names, gal_rel, image_root, is_cover=False)
            if ref and ref.source_path not in seen_paths:
                image_refs.append(ref)
                seen_paths.add(ref.source_path)

        created = _parse_dt(r.created_at)

        products.append(Product(
            source_id=r.entity_id,
            sku=(r.sku or "")[:64],
            model=(r.sku or "")[:64],
            price=float(price_raw),
            quantity=int(float(qty_raw)),
            weight=float(weight_raw),
            status=status_enabled,
            sort_order=0,
            manufacturer_source_id=None,
            ean="", upc="", isbn="", mpn="",
            minimum=1,
            width=0.0, height=0.0, depth=0.0,
            date_added=created,
            category_source_ids=cat_links.get(r.entity_id, []),
            names=names,
            descriptions=descriptions,
            meta_titles=meta_titles,
            meta_descriptions=meta_descs,
            meta_keywords=meta_kws,
            images=image_refs,
        ))

    return products


def _load_eav(conn) -> dict[int, dict[int, dict[str, str]]]:
    out: dict[int, dict[int, dict[str, str]]] = {}
    for tbl, mapping in (
        ("catalog_product_entity_varchar", PROD_ATTR_VARCHAR),
        ("catalog_product_entity_text",    PROD_ATTR_TEXT),
        ("catalog_product_entity_decimal", PROD_ATTR_DECIMAL),
        ("catalog_product_entity_int",     PROD_ATTR_INT),
    ):
        ids = ",".join(str(i) for i in mapping)
        rows = conn.execute(text(
            f"SELECT entity_id, attribute_id, store_id, value "
            f"FROM {tbl} WHERE attribute_id IN ({ids})"
        )).fetchall()
        for r in rows:
            code = mapping[r.attribute_id]
            out.setdefault(r.entity_id, {}).setdefault(r.store_id, {})[code] = r.value
    return out


def _pick(per_store: dict[int, dict[str, str]], store_id: int, code: str, default=None):
    if store_id in per_store and code in per_store[store_id]:
        return per_store[store_id][code]
    if 0 in per_store and code in per_store[0]:
        return per_store[0][code]
    return default


def _load_stock(conn) -> dict[int, str]:
    out: dict[int, str] = {}
    for r in conn.execute(text(
        "SELECT product_id, qty FROM cataloginventory_stock_item WHERE stock_id = 1"
    )):
        out[r.product_id] = str(r.qty)
    return out


def _load_category_links(conn) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r in conn.execute(text(
        "SELECT product_id, category_id FROM catalog_category_product"
    )):
        out.setdefault(r.product_id, []).append(r.category_id)
    return out


def _load_images(conn) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    rows = conn.execute(text(
        "SELECT g.entity_id, g.value, v.position, v.disabled "
        "FROM catalog_product_entity_media_gallery g "
        "LEFT JOIN catalog_product_entity_media_gallery_value v "
        "  ON v.value_id = g.value_id AND v.store_id = 0 "
        "WHERE g.value IS NOT NULL AND g.value <> '' "
        "ORDER BY g.entity_id, v.position"
    )).fetchall()
    for r in rows:
        if r.disabled:
            continue
        out.setdefault(r.entity_id, []).append(r.value)
    return out


def _make_image_ref(
    pid: int, names: dict[int, str], rel: str,
    image_root: Optional[Path], is_cover: bool,
) -> Optional[ImageRef]:
    if not image_root or not rel or rel == "no_selection":
        return None
    src = image_root / "media" / "catalog" / "product" / rel.lstrip("/")
    if not src.exists():
        return None
    first_name = next(iter(names.values()), f"prod_{pid}")
    safe = first_name.lower().replace(" ", "-")[:50]
    ext = src.suffix.lower() or ".jpg"
    suffix = "" if is_cover else f"-{hash(rel) & 0xFFFF:04x}"
    return ImageRef(
        source_path=str(src),
        dest_filename=f"product/{safe}-{pid}{suffix}{ext}",
        is_cover=is_cover,
    )


def _parse_dt(v) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except Exception:
        return None
