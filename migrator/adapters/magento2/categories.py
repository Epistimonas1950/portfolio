"""Read Magento 2.x categories into canonical Category objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Category, ImageRef
from migrator.adapters.magento2._eav import load_attribute_map, attrs_by_backend

# Attribute codes we care about. IDs are resolved per-install at runtime.
CAT_CODES = [
    "name", "description", "image",
    "meta_title", "meta_keywords", "meta_description",
    "is_active",
]


def load_categories(
    engine: Engine,
    prefix: str,                       # unused for M2 (no prefix)
    lang_map: dict[int, int],          # m2_store_id → oc_lang_id (with store_id=0 admin fallback)
    image_root: Optional[Path] = None, # path to Magento root (for pub/media/catalog/category/)
) -> list[Category]:
    with engine.connect() as conn:
        attr_map = load_attribute_map(conn, "catalog_category")
        per_backend = attrs_by_backend(attr_map, CAT_CODES)

        # M2 default store-root is entity_id=2 (M1 was 3).
        cat_rows = conn.execute(text(
            "SELECT entity_id, parent_id, position, level "
            "FROM catalog_category_entity "
            "WHERE path LIKE '1/2/%' "
            "ORDER BY parent_id, position"
        )).fetchall()

        eav: dict[int, dict[int, dict[int, str]]] = {}
        for btype, attrs in per_backend.items():
            if not attrs:
                continue
            ids = ",".join(str(i) for i in attrs)
            tbl = f"catalog_category_entity_{btype}"
            rows = conn.execute(text(
                f"SELECT entity_id, attribute_id, store_id, value "
                f"FROM {tbl} WHERE attribute_id IN ({ids})"
            )).fetchall()
            for r in rows:
                eav.setdefault(r.entity_id, {}).setdefault(r.store_id, {})[r.attribute_id] = r.value

    # Reverse lookup: attribute_id → code
    id_to_code: dict[int, str] = {}
    for attrs in per_backend.values():
        id_to_code.update(attrs)

    def pick(eid: int, store_id: int, code: str, default=None):
        # Find the attribute_id for this code; if missing in attr_map, skip.
        aid = next((i for i, c in id_to_code.items() if c == code), None)
        if aid is None:
            return default
        slots = eav.get(eid, {})
        if store_id in slots and aid in slots[store_id]:
            return slots[store_id][aid]
        if 0 in slots and aid in slots[0]:
            return slots[0][aid]
        return default

    categories: list[Category] = []
    for r in cat_rows:
        names, descriptions = {}, {}
        meta_titles, meta_descs, meta_kws = {}, {}, {}

        for m2_store, oc_lang in lang_map.items():
            name = pick(r.entity_id, m2_store, "name")
            if not name:
                continue
            names[oc_lang] = name
            descriptions[oc_lang] = pick(r.entity_id, m2_store, "description", "") or ""
            meta_titles[oc_lang] = pick(r.entity_id, m2_store, "meta_title", "") or ""
            meta_descs[oc_lang] = pick(r.entity_id, m2_store, "meta_description", "") or ""
            meta_kws[oc_lang] = pick(r.entity_id, m2_store, "meta_keywords", "") or ""

        if not names:
            continue

        is_active_raw = pick(r.entity_id, 0, "is_active", "1")
        status = bool(int(is_active_raw)) if is_active_raw is not None else True

        # parent_id == 2 means it's a child of the store-root → top-level in OC.
        parent_id = r.parent_id if r.parent_id and r.parent_id != 2 else None

        image_rel = pick(r.entity_id, 0, "image")
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
    # M2 stores category images under pub/media/catalog/category/
    src = image_root / "pub" / "media" / "catalog" / "category" / rel
    if not src.exists():
        # Fallback for installs that don't use pub/ (rare M2 setups)
        src = image_root / "media" / "catalog" / "category" / rel
        if not src.exists():
            return None
    first_name = next(iter(names.values()), f"cat_{cat_id}")
    safe = first_name.lower().replace(" ", "-")[:50]
    ext = src.suffix.lower() or ".jpg"
    return ImageRef(source_path=str(src), dest_filename=f"category/{safe}-{cat_id}{ext}")
