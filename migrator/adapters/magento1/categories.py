"""Read Magento 1.x categories into canonical Category objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Category, ImageRef

# EAV attribute IDs for entity_type_id=9 (catalog_category) on this M1 install.
CAT_ATTR = {
    "name": 111,             # varchar
    "description": 112,      # text
    "image": 113,            # varchar
    "meta_title": 114,       # varchar
    "meta_keywords": 115,    # text
    "meta_description": 116, # text
    "is_active": 119,        # int
}


def load_categories(
    engine: Engine,
    prefix: str,                       # unused for M1 (no prefix)
    lang_map: dict[int, int],          # m1_store_id → oc_lang_id (with store_id=0 admin fallback)
    image_root: Optional[Path] = None, # path to Magento root (for media/catalog/category/)
) -> list[Category]:
    with engine.connect() as conn:
        # Real categories only: descended from default store root (entity_id=3).
        # Skips the M1 internal root (1), per-store roots (3, 105), and stray orphans.
        cat_rows = conn.execute(text(
            "SELECT entity_id, parent_id, position, level "
            "FROM catalog_category_entity "
            "WHERE path LIKE '1/3/%' "
            "ORDER BY parent_id, position"
        )).fetchall()

        varchar_rows = conn.execute(text(
            "SELECT entity_id, attribute_id, store_id, value "
            "FROM catalog_category_entity_varchar "
            "WHERE attribute_id IN (111, 113, 114)"
        )).fetchall()
        text_rows = conn.execute(text(
            "SELECT entity_id, attribute_id, store_id, value "
            "FROM catalog_category_entity_text "
            "WHERE attribute_id IN (112, 115, 116)"
        )).fetchall()
        int_rows = conn.execute(text(
            "SELECT entity_id, attribute_id, store_id, value "
            "FROM catalog_category_entity_int "
            "WHERE attribute_id = 119"
        )).fetchall()

    # Pivot to: eav[entity_id][store_id][attribute_id] = value
    eav: dict[int, dict[int, dict[int, str]]] = {}
    for r in (*varchar_rows, *text_rows, *int_rows):
        eav.setdefault(r.entity_id, {}).setdefault(r.store_id, {})[r.attribute_id] = r.value

    def pick(eid: int, store_id: int, attr_id: int, default=None):
        slots = eav.get(eid, {})
        if store_id in slots and attr_id in slots[store_id]:
            return slots[store_id][attr_id]
        if 0 in slots and attr_id in slots[0]:
            return slots[0][attr_id]
        return default

    categories: list[Category] = []
    for r in cat_rows:
        names, descriptions = {}, {}
        meta_titles, meta_descs, meta_kws = {}, {}, {}

        for m1_store, oc_lang in lang_map.items():
            name = pick(r.entity_id, m1_store, CAT_ATTR["name"])
            if not name:
                continue
            names[oc_lang] = name
            descriptions[oc_lang] = pick(r.entity_id, m1_store, CAT_ATTR["description"], "") or ""
            meta_titles[oc_lang] = pick(r.entity_id, m1_store, CAT_ATTR["meta_title"], "") or ""
            meta_descs[oc_lang] = pick(r.entity_id, m1_store, CAT_ATTR["meta_description"], "") or ""
            meta_kws[oc_lang] = pick(r.entity_id, m1_store, CAT_ATTR["meta_keywords"], "") or ""

        if not names:
            continue

        is_active_raw = pick(r.entity_id, 0, CAT_ATTR["is_active"], "1")
        status = bool(int(is_active_raw)) if is_active_raw is not None else True

        # parent_id == 3 means it's a child of the store-root → top-level in OC.
        parent_id = r.parent_id if r.parent_id and r.parent_id != 3 else None

        image_rel = pick(r.entity_id, 0, CAT_ATTR["image"])
        image = _resolve_image(r.entity_id, names, image_rel, image_root)

        categories.append(Category(
            source_id=r.entity_id,
            parent_source_id=parent_id,
            sort_order=r.position,
            status=status,
            names=names,
            descriptions=descriptions,
            meta_titles=meta_titles,
            meta_descriptions=meta_descs,
            meta_keywords=meta_kws,
            image=image,
        ))

    return categories


def _resolve_image(
    cat_id: int,
    names: dict[int, str],
    image_value: Optional[str],
    image_root: Optional[Path],
) -> Optional[ImageRef]:
    if not image_root or not image_value or image_value == "no_selection":
        return None
    rel = image_value.lstrip("/")
    src = image_root / "media" / "catalog" / "category" / rel
    if not src.exists():
        return None
    first_name = next(iter(names.values()), f"cat_{cat_id}")
    safe = first_name.lower().replace(" ", "-")[:50]
    ext = src.suffix.lower() or ".jpg"
    return ImageRef(source_path=str(src), dest_filename=f"category/{safe}-{cat_id}{ext}")
