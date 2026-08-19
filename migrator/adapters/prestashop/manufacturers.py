"""Read PrestaShop manufacturers into canonical Manufacturer objects."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import ImageRef, Manufacturer


def load_manufacturers(
    engine: Engine,
    prefix: str,
    image_root: Optional[Path] = None,
) -> list[Manufacturer]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT id_manufacturer, name, active "
            f"FROM `{prefix}manufacturer` "
            f"WHERE active = 1 "
            f"ORDER BY id_manufacturer"
        )).fetchall()

    out: list[Manufacturer] = []
    for r in rows:
        out.append(Manufacturer(
            source_id=r.id_manufacturer,
            name=r.name or f"Manufacturer {r.id_manufacturer}",
            sort_order=0,
            image=_resolve_logo(r.id_manufacturer, r.name or "", image_root),
        ))
    return out


def _resolve_logo(mid: int, name: str, image_root: Optional[Path]) -> Optional[ImageRef]:
    if not image_root:
        return None
    src = image_root / "img" / "m" / f"{mid}.jpg"
    if not src.exists():
        return None
    slug = re.sub(r"[^A-Za-z0-9 ]", "", name)
    slug = re.sub(r"\s+", "-", slug.strip().lower())[:50].strip("-") or f"manu-{mid}"
    return ImageRef(
        source_path=str(src),
        dest_filename=f"manufacturer/{slug}-{mid}.jpg",
    )
