"""Write canonical Category objects into an OpenCart 3.x database."""
from __future__ import annotations
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Category
from migrator.pipeline.id_map import (
    IdMapStore, ENTITY_CATEGORY, chunked, DEFAULT_BATCH_SIZE,
)


def write_categories(
    engine: Engine,
    categories: list[Category],
    prefix: str,
    oc_image_dir: Optional[Path] = None,
    store_id: int = 0,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_every: int = 25,
    id_map_store: Optional[IdMapStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, int]:
    """Insert categories in parent-first order. Returns {source_id: oc_category_id}.

    Idempotent: if `id_map_store` already has mappings for some categories,
    those are skipped and only new ones are inserted. Commits per batch so a
    crash mid-run only loses up to `batch_size` rows of progress.
    """
    existing = id_map_store.load(ENTITY_CATEGORY) if id_map_store else {}
    id_map: dict[int, int] = dict(existing)

    ordered = _topological_sort(categories)
    todo = [c for c in ordered if c.source_id not in existing]

    if log_cb and existing:
        log_cb(f"  • Already migrated: {len(existing)} categories — skipping")
    if not todo:
        if progress_cb:
            progress_cb(len(existing), len(existing) or 1)
        return id_map

    # For resume scenarios, ancestors of new categories may have been written
    # in a prior run, so their path_map entries aren't in memory. Rebuild from
    # the existing oc_category_path rows.
    path_map = _load_path_map(engine, prefix, existing.values()) if existing else {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total = len(todo)
    processed = 0

    for chunk in chunked(todo, batch_size):
        with engine.begin() as conn:
            for cat in chunk:
                parent_oc_id = id_map.get(cat.parent_source_id, 0) if cat.parent_source_id else 0
                image_val = _copy_image(cat, oc_image_dir)
                # top=1 only for root-level categories — that's what the OC default
                # header renders in the nav bar. Children show as flyouts via parent_id.
                top_flag = 1 if parent_oc_id == 0 else 0

                result = conn.execute(text(
                    f"INSERT INTO `{prefix}category` "
                    f"(image, parent_id, top, `column`, sort_order, status, date_added, date_modified) "
                    f"VALUES (:img, :parent, :top, 1, :sort, :status, :now, :now)"
                ), {"img": image_val, "parent": parent_oc_id, "top": top_flag,
                    "sort": cat.sort_order, "status": int(cat.status), "now": now})
                oc_id = result.lastrowid
                id_map[cat.source_id] = oc_id

                for lang_id, name in cat.names.items():
                    conn.execute(text(
                        f"INSERT INTO `{prefix}category_description` "
                        f"(category_id, language_id, name, description, "
                        f"meta_title, meta_description, meta_keyword) "
                        f"VALUES (:cid, :lid, :name, :desc, :mt, :md, :mk)"
                    ), {
                        "cid": oc_id, "lid": lang_id, "name": name[:255],
                        "desc": cat.descriptions.get(lang_id, ""),
                        "mt": cat.meta_titles.get(lang_id, "")[:255],
                        "md": cat.meta_descriptions.get(lang_id, "")[:255],
                        "mk": cat.meta_keywords.get(lang_id, "")[:255],
                    })

                conn.execute(text(
                    f"INSERT INTO `{prefix}category_to_store` (category_id, store_id) "
                    f"VALUES (:cid, :sid)"
                ), {"cid": oc_id, "sid": store_id})

                parent_path = path_map.get(parent_oc_id, []) if parent_oc_id else []
                my_path = parent_path + [oc_id]
                path_map[oc_id] = my_path

                for level, path_id in enumerate(my_path):
                    conn.execute(text(
                        f"INSERT INTO `{prefix}category_path` (category_id, path_id, level) "
                        f"VALUES (:cid, :pid, :lv)"
                    ), {"cid": oc_id, "pid": path_id, "lv": level})

                if id_map_store:
                    id_map_store.record(conn, ENTITY_CATEGORY, cat.source_id, oc_id)

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % log_every == 0 or processed == total):
                    label = next(iter(cat.names.values()), f"id={cat.source_id}")
                    log_cb(f"  ✓ [{processed}/{total}] last: {label}")

    return id_map


def _copy_image(cat: Category, oc_image_dir: Optional[Path]) -> str:
    if not cat.image or not oc_image_dir:
        return ""
    oc_path = f"catalog/{cat.image.dest_filename}"
    dest = oc_image_dir / oc_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(cat.image.source_path, dest)
        return oc_path
    except OSError:
        return ""


def _topological_sort(categories: list[Category]) -> list[Category]:
    """Return categories sorted so every parent appears before its children."""
    by_id = {c.source_id: c for c in categories}
    visited: set[int] = set()
    result: list[Category] = []

    def visit(cat: Category) -> None:
        if cat.source_id in visited:
            return
        visited.add(cat.source_id)
        if cat.parent_source_id and cat.parent_source_id in by_id:
            visit(by_id[cat.parent_source_id])
        result.append(cat)

    for cat in categories:
        visit(cat)
    return result


def _load_path_map(engine: Engine, prefix: str, oc_ids) -> dict[int, list[int]]:
    """Rebuild {oc_category_id → [ancestor_oc_ids..., self]} from oc_category_path
    for the given category ids. Used on resume so newly-inserted children inherit
    their already-migrated parents' full path.
    """
    oc_ids = list(oc_ids)
    if not oc_ids:
        return {}
    stmt = text(
        f"SELECT category_id, path_id, level FROM `{prefix}category_path` "
        f"WHERE category_id IN :ids ORDER BY category_id, level"
    ).bindparams(bindparam("ids", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"ids": oc_ids}).fetchall()

    path_map: dict[int, list[int]] = {}
    for r in rows:
        path_map.setdefault(int(r.category_id), []).append(int(r.path_id))
    return path_map
