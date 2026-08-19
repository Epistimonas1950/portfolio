"""Read PrestaShop products into canonical Product objects."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import ImageRef, Product


def load_products(
    engine: Engine,
    prefix: str,
    lang_map: dict[int, int],
    image_root: Optional[Path] = None,
    only_active: bool = True,
) -> list[Product]:
    """Load PrestaShop products and return canonical Product objects."""
    where_active = "WHERE p.active = 1" if only_active else ""

    with engine.connect() as conn:
        cat_links = _load_category_links(conn, prefix)
        images_idx = _load_image_index(conn, prefix)
        stock_idx = _load_stock(conn, prefix)
        lang_idx = _load_language_data(conn, prefix)

        rows = conn.execute(text(
            f"SELECT p.id_product, p.id_manufacturer, p.reference, p.supplier_reference, "
            f"p.ean13, p.upc, p.isbn, p.mpn, p.price, p.weight, p.width, p.height, p.depth, "
            f"p.minimal_quantity, p.active, p.date_add "
            f"FROM `{prefix}product` p "
            f"{where_active} "
            f"ORDER BY p.id_product"
        )).fetchall()

    return [
        _to_canonical(r, cat_links, images_idx, stock_idx, lang_idx, lang_map, image_root)
        for r in rows
    ]


def _load_category_links(conn, prefix: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for r in conn.execute(text(
        f"SELECT id_product, id_category FROM `{prefix}category_product`"
    )):
        out.setdefault(r.id_product, []).append(r.id_category)
    return out


def _load_image_index(conn, prefix: str) -> dict[int, list[int]]:
    """Return {product_id: [image_id, ...]} with the cover image always first."""
    out: dict[int, list[int]] = {}
    rows = conn.execute(text(
        f"SELECT id_product, id_image, cover, position "
        f"FROM `{prefix}image` "
        f"ORDER BY id_product, cover DESC, position"
    )).fetchall()
    for r in rows:
        out.setdefault(r.id_product, []).append(r.id_image)
    return out


def _load_stock(conn, prefix: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for r in conn.execute(text(
        f"SELECT id_product, SUM(quantity) AS qty "
        f"FROM `{prefix}stock_available` "
        f"WHERE id_product_attribute = 0 "
        f"GROUP BY id_product"
    )):
        out[r.id_product] = int(r.qty or 0)
    return out


def _load_language_data(conn, prefix: str) -> dict[int, dict[int, dict]]:
    out: dict[int, dict[int, dict]] = {}
    for r in conn.execute(text(
        f"SELECT id_product, id_lang, name, description, description_short, "
        f"meta_title, meta_description, meta_keywords "
        f"FROM `{prefix}product_lang`"
    )):
        parts = []
        if r.description_short:
            parts.append(r.description_short)
        if r.description:
            parts.append(r.description)
        out.setdefault(r.id_product, {})[r.id_lang] = {
            "name": r.name or "",
            "description": "\n\n".join(parts),
            "meta_title": r.meta_title or "",
            "meta_description": r.meta_description or "",
            "meta_keywords": r.meta_keywords or "",
        }
    return out


def _to_canonical(
    r,
    cat_links: dict[int, list[int]],
    images_idx: dict[int, list[int]],
    stock_idx: dict[int, int],
    lang_idx: dict[int, dict[int, dict]],
    lang_map: dict[int, int],
    image_root: Optional[Path],
) -> Product:
    pid = r.id_product
    model = (r.reference or r.supplier_reference or f"PS-{pid}").strip()[:64]
    sku = (r.supplier_reference or r.reference or f"PS-{pid}").strip()[:64]

    names, descriptions, meta_titles, meta_descs, meta_kws = {}, {}, {}, {}, {}
    for ps_lang, oc_lang in lang_map.items():
        ld = lang_idx.get(pid, {}).get(ps_lang)
        if ld:
            names[oc_lang] = ld["name"]
            descriptions[oc_lang] = ld["description"]
            meta_titles[oc_lang] = ld["meta_title"]
            meta_descs[oc_lang] = ld["meta_description"]
            meta_kws[oc_lang] = ld["meta_keywords"]

    slug = _slugify(next(iter(names.values()), ""), fallback=f"product-{pid}")

    images: list[ImageRef] = []
    for idx, id_image in enumerate(images_idx.get(pid, [])):
        src = _ps_image_path(id_image, image_root) if image_root else None
        if image_root and src and not src.exists():
            continue
        images.append(ImageRef(
            source_path=str(src) if src else "",
            dest_filename=f"product/{slug}-{pid}-{id_image}.jpg",
            is_cover=(idx == 0),
        ))

    return Product(
        source_id=pid,
        sku=sku,
        model=model,
        price=float(r.price or 0),
        quantity=stock_idx.get(pid, 0),
        weight=float(r.weight or 0),
        status=bool(r.active),
        manufacturer_source_id=r.id_manufacturer or None,
        ean=(r.ean13 or "")[:14],
        upc=(r.upc or "")[:12],
        isbn=(r.isbn or "")[:17],
        mpn=(r.mpn or "")[:64],
        minimum=int(r.minimal_quantity or 1),
        width=float(r.width or 0),
        height=float(r.height or 0),
        depth=float(r.depth or 0),
        date_added=r.date_add,
        category_source_ids=cat_links.get(pid, []),
        names=names,
        descriptions=descriptions,
        meta_titles=meta_titles,
        meta_descriptions=meta_descs,
        meta_keywords=meta_kws,
        images=images,
    )


def _slugify(name: str, fallback: str) -> str:
    """ASCII slug, lowercase, max 50 chars. Falls back if name has no ASCII letters."""
    ascii_only = re.sub(r"[^A-Za-z0-9 ]", "", name)
    slug = re.sub(r"\s+", "-", ascii_only.strip().lower())[:50].strip("-")
    return slug or fallback


def _ps_image_path(id_image: int, image_root: Path) -> Optional[Path]:
    """PrestaShop splits image IDs into digit-folders: 12345 -> 1/2/3/4/5/12345.jpg"""
    digits = list(str(id_image))
    return image_root / "img" / "p" / Path(*digits) / f"{id_image}.jpg"
