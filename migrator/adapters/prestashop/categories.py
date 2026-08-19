"""Read PrestaShop categories into canonical Category objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Category, ImageRef


def make_engine(host: str, port: int, user: str, password: str, db: str) -> Engine:
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def load_categories(
    engine: Engine,
    prefix: str,
    lang_map: dict[int, int],           # ps_lang_id → oc_lang_id
    image_root: Optional[Path] = None,  # path to PrestaShop root (for img/c/)
) -> list[Category]:
    with engine.connect() as conn:
        cat_rows = conn.execute(text(
            f"SELECT id_category, id_parent, position, active "
            f"FROM `{prefix}category` "
            f"WHERE id_category > 1 "   # skip PS internal root (id=1)
            f"ORDER BY id_parent, position"
        )).fetchall()

        lang_rows = conn.execute(text(
            f"SELECT id_category, id_lang, name, description, "
            f"meta_title, meta_description, meta_keywords "
            f"FROM `{prefix}category_lang` "
            f"WHERE id_category > 1"
        )).fetchall()

    # Index language data by category
    lang_index: dict[int, dict[int, dict]] = {}
    for r in lang_rows:
        lang_index.setdefault(r.id_category, {})[r.id_lang] = {
            "name": r.name or "",
            "description": r.description or "",
            "meta_title": r.meta_title or "",
            "meta_description": r.meta_description or "",
            "meta_keywords": r.meta_keywords or "",
        }

    categories: list[Category] = []
    for r in cat_rows:
        cat_langs = lang_index.get(r.id_category, {})

        names, descriptions, meta_titles, meta_descs, meta_kws = {}, {}, {}, {}, {}
        for ps_lang, oc_lang in lang_map.items():
            ld = cat_langs.get(ps_lang)
            if ld:
                names[oc_lang] = ld["name"]
                descriptions[oc_lang] = ld["description"]
                meta_titles[oc_lang] = ld["meta_title"]
                meta_descs[oc_lang] = ld["meta_description"]
                meta_kws[oc_lang] = ld["meta_keywords"]

        if not names:
            continue  # no matching language — skip

        image = _resolve_image(r.id_category, names, image_root)
        parent_id = r.id_parent if r.id_parent and r.id_parent > 1 else None

        categories.append(Category(
            source_id=r.id_category,
            parent_source_id=parent_id,
            sort_order=r.position,
            status=bool(r.active),
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
    image_root: Optional[Path],
) -> Optional[ImageRef]:
    if not image_root:
        return None
    src = image_root / "img" / "c" / f"{cat_id}.jpg"
    if not src.exists():
        return None
    first_name = next(iter(names.values()), f"cat_{cat_id}")
    safe = first_name.lower().replace(" ", "-")[:50]
    return ImageRef(source_path=str(src), dest_filename=f"category/{safe}-{cat_id}.jpg")
