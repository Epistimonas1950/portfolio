"""Read WooCommerce product categories into canonical Category objects."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Category, ImageRef


def load_categories(
    engine: Engine,
    prefix: str,
    lang_map: dict[int, int],
    image_root: Optional[Path] = None,
) -> list[Category]:
    with engine.connect() as conn:
        cat_rows = conn.execute(text(
            f"SELECT t.term_id, t.name, t.slug, "
            f"       tt.parent, tt.description "
            f"FROM `{prefix}term_taxonomy` tt "
            f"JOIN `{prefix}terms` t ON t.term_id = tt.term_id "
            f"WHERE tt.taxonomy = 'product_cat' "
            f"ORDER BY tt.parent, t.name"
        )).fetchall()

        meta_rows = conn.execute(text(
            f"SELECT term_id, meta_key, meta_value FROM `{prefix}termmeta` "
            f"WHERE meta_key IN ('order', 'thumbnail_id')"
        )).fetchall()
        meta: dict[int, dict[str, str]] = {}
        for r in meta_rows:
            meta.setdefault(r.term_id, {})[r.meta_key] = r.meta_value

        attachment_ids = {
            int(v["thumbnail_id"])
            for v in meta.values()
            if v.get("thumbnail_id", "").isdigit() and int(v["thumbnail_id"]) > 0
        }
        attachments = _load_attachment_paths(conn, prefix, attachment_ids)

    oc_lang = next(iter(lang_map.values()), 1) if lang_map else 1

    categories: list[Category] = []
    for r in cat_rows:
        m = meta.get(r.term_id, {})
        sort_order = int(m["order"]) if m.get("order", "").lstrip("-").isdigit() else 0
        parent = r.parent if r.parent and r.parent > 0 else None

        thumb_id = int(m["thumbnail_id"]) if m.get("thumbnail_id", "").isdigit() else 0
        image = None
        if thumb_id and thumb_id in attachments and image_root:
            rel = attachments[thumb_id]
            src = image_root / "wp-content" / "uploads" / rel
            if src.exists():
                safe = (r.slug or r.name or f"cat_{r.term_id}").lower().replace(" ", "-")[:50]
                ext = src.suffix.lower() or ".jpg"
                image = ImageRef(
                    source_path=str(src),
                    dest_filename=f"category/{safe}-{r.term_id}{ext}",
                )

        categories.append(Category(
            source_id=r.term_id,
            parent_source_id=parent,
            sort_order=sort_order,
            status=True,
            names={oc_lang: r.name or f"category_{r.term_id}"},
            descriptions={oc_lang: r.description or ""},
            meta_titles={oc_lang: ""},
            meta_descriptions={oc_lang: ""},
            meta_keywords={oc_lang: ""},
            image=image,
        ))

    return categories


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
