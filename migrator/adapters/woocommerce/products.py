"""Read WooCommerce products into canonical Product objects."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import ImageRef, Product

PRODUCT_META_KEYS = (
    "_sku", "_price", "_regular_price", "_sale_price",
    "_stock", "_stock_status", "_manage_stock",
    "_weight", "_length", "_width", "_height",
    "_thumbnail_id", "_product_image_gallery",
)


def load_products(
    engine: Engine,
    prefix: str,
    lang_map: dict[int, int],
    image_root: Optional[Path] = None,
    only_active: bool = True,
) -> list[Product]:
    oc_lang = next(iter(lang_map.values()), 1) if lang_map else 1
    status_filter = " AND post_status = 'publish'" if only_active else ""

    with engine.connect() as conn:
        prod_rows = conn.execute(text(
            f"SELECT ID, post_title, post_name, post_status, "
            f"       post_content, post_excerpt, post_date "
            f"FROM `{prefix}posts` "
            f"WHERE post_type = 'product'{status_filter} "
            f"ORDER BY ID"
        )).fetchall()
        prod_ids = [r.ID for r in prod_rows]
        if not prod_ids:
            return []

        meta = _load_meta(conn, prefix, prod_ids)
        cat_links = _load_category_links(conn, prefix, prod_ids)
        attachment_ids = _collect_attachment_ids(meta)
        attachments = _load_attachment_paths(conn, prefix, attachment_ids)

    products: list[Product] = []
    for r in prod_rows:
        m = meta.get(r.ID, {})

        price = _to_float(m.get("_price") or m.get("_regular_price") or "0")
        weight = _to_float(m.get("_weight") or "0")
        sku = (m.get("_sku") or "")[:64]
        qty = _to_int(m.get("_stock") or "0")
        stock_status = m.get("_stock_status") or "instock"

        image_refs: list[ImageRef] = []
        seen = set()
        cover_id = _to_int(m.get("_thumbnail_id") or "0")
        if cover_id and cover_id in attachments:
            ref = _make_image_ref(
                r.ID, r.post_name or r.post_title or f"prod_{r.ID}",
                attachments[cover_id], image_root, is_cover=True,
            )
            if ref:
                image_refs.append(ref)
                seen.add(ref.source_path)
        for aid_str in (m.get("_product_image_gallery") or "").split(","):
            aid = _to_int(aid_str.strip())
            if aid and aid in attachments:
                ref = _make_image_ref(
                    r.ID, r.post_name or r.post_title or f"prod_{r.ID}",
                    attachments[aid], image_root, is_cover=False,
                )
                if ref and ref.source_path not in seen:
                    image_refs.append(ref)
                    seen.add(ref.source_path)

        short = (r.post_excerpt or "").strip()
        long_ = (r.post_content or "").strip()
        desc = (short + "\n" + long_).strip() if short and long_ else (long_ or short)

        products.append(Product(
            source_id=r.ID,
            sku=sku,
            model=sku or f"prod-{r.ID}",
            price=price,
            quantity=max(qty, 0) if stock_status == "instock" else 0,
            weight=weight,
            status=(r.post_status == "publish"),
            sort_order=0,
            manufacturer_source_id=None,
            ean="", upc="", isbn="", mpn="",
            minimum=1,
            width=_to_float(m.get("_width") or "0"),
            height=_to_float(m.get("_height") or "0"),
            depth=_to_float(m.get("_length") or "0"),
            date_added=_parse_dt(r.post_date),
            category_source_ids=cat_links.get(r.ID, []),
            names={oc_lang: r.post_title or f"product_{r.ID}"},
            descriptions={oc_lang: desc},
            meta_titles={oc_lang: ""},
            meta_descriptions={oc_lang: ""},
            meta_keywords={oc_lang: ""},
            images=image_refs,
        ))

    return products


def _load_meta(conn, prefix: str, ids: list[int]) -> dict[int, dict[str, str]]:
    placeholders = ",".join(str(i) for i in ids)
    key_list = ",".join(f"'{k}'" for k in PRODUCT_META_KEYS)
    out: dict[int, dict[str, str]] = {}
    for r in conn.execute(text(
        f"SELECT post_id, meta_key, meta_value FROM `{prefix}postmeta` "
        f"WHERE post_id IN ({placeholders}) AND meta_key IN ({key_list})"
    )):
        out.setdefault(r.post_id, {})[r.meta_key] = r.meta_value
    return out


def _load_category_links(conn, prefix: str, ids: list[int]) -> dict[int, list[int]]:
    placeholders = ",".join(str(i) for i in ids)
    out: dict[int, list[int]] = {}
    for r in conn.execute(text(
        f"SELECT tr.object_id, tt.term_id "
        f"FROM `{prefix}term_relationships` tr "
        f"JOIN `{prefix}term_taxonomy` tt ON tt.term_taxonomy_id = tr.term_taxonomy_id "
        f"WHERE tr.object_id IN ({placeholders}) AND tt.taxonomy = 'product_cat'"
    )):
        out.setdefault(r.object_id, []).append(r.term_id)
    return out


def _collect_attachment_ids(meta: dict[int, dict[str, str]]) -> set[int]:
    out: set[int] = set()
    for m in meta.values():
        tid = _to_int(m.get("_thumbnail_id") or "0")
        if tid:
            out.add(tid)
        for aid_str in (m.get("_product_image_gallery") or "").split(","):
            aid = _to_int(aid_str.strip())
            if aid:
                out.add(aid)
    return out


def _load_attachment_paths(conn, prefix: str, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    placeholders = ",".join(str(i) for i in ids)
    out: dict[int, str] = {}
    for r in conn.execute(text(
        f"SELECT post_id, meta_value FROM `{prefix}postmeta` "
        f"WHERE meta_key = '_wp_attached_file' AND post_id IN ({placeholders})"
    )):
        out[r.post_id] = r.meta_value or ""
    return out


def _make_image_ref(
    pid: int, slug: str, rel: str,
    image_root: Optional[Path], is_cover: bool,
) -> Optional[ImageRef]:
    if not image_root or not rel:
        return None
    src = image_root / "wp-content" / "uploads" / rel
    if not src.exists():
        return None
    safe = slug.lower().replace(" ", "-")[:50]
    ext = src.suffix.lower() or ".jpg"
    suffix = "" if is_cover else f"-{hash(rel) & 0xFFFF:04x}"
    return ImageRef(
        source_path=str(src),
        dest_filename=f"product/{safe}-{pid}{suffix}{ext}",
        is_cover=is_cover,
    )


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
