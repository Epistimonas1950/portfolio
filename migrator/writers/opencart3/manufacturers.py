"""Write canonical Manufacturer objects into OpenCart 3.x."""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import Manufacturer
from migrator.pipeline.id_map import (
    IdMapStore, ENTITY_MANUFACTURER, chunked, DEFAULT_BATCH_SIZE,
)


def write_manufacturers(
    engine: Engine,
    manufacturers: list[Manufacturer],
    prefix: str,
    oc_image_dir: Optional[Path] = None,
    store_id: int = 0,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_every: int = 25,
    id_map_store: Optional[IdMapStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[int, int]:
    """Insert manufacturers. Returns {source_id: oc_manufacturer_id}.

    Idempotent: skips manufacturers already present in `id_map_store`.
    Commits per batch so a crash mid-run only loses up to `batch_size` rows.
    """
    existing = id_map_store.load(ENTITY_MANUFACTURER) if id_map_store else {}
    id_map: dict[int, int] = dict(existing)

    todo = [m for m in manufacturers if m.source_id not in existing]
    if log_cb and existing:
        log_cb(f"  • Already migrated: {len(existing)} manufacturers — skipping")
    if not todo:
        if progress_cb:
            progress_cb(len(existing), len(existing) or 1)
        return id_map

    total = len(todo)
    processed = 0

    for chunk in chunked(todo, batch_size):
        with engine.begin() as conn:
            for m in chunk:
                image_val = _copy_image(m, oc_image_dir)
                result = conn.execute(text(
                    f"INSERT INTO `{prefix}manufacturer` (name, image, sort_order) "
                    f"VALUES (:name, :img, :sort)"
                ), {"name": m.name, "img": image_val, "sort": m.sort_order})
                oc_id = result.lastrowid
                id_map[m.source_id] = oc_id

                conn.execute(text(
                    f"INSERT INTO `{prefix}manufacturer_to_store` (manufacturer_id, store_id) "
                    f"VALUES (:mid, :sid)"
                ), {"mid": oc_id, "sid": store_id})

                if id_map_store:
                    id_map_store.record(conn, ENTITY_MANUFACTURER, m.source_id, oc_id)

                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % log_every == 0 or processed == total):
                    log_cb(f"  ✓ [{processed}/{total}] last: {m.name}")

    return id_map


def _copy_image(m: Manufacturer, oc_image_dir: Optional[Path]) -> str:
    if not m.image or not oc_image_dir:
        return ""
    oc_path = f"catalog/{m.image.dest_filename}"
    dest = oc_image_dir / oc_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(m.image.source_path, dest)
        return oc_path
    except OSError:
        return ""
