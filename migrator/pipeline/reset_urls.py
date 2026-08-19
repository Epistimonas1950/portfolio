"""Generate password-reset codes for migrated customers and export a CSV
of one-click reset links suitable for a bulk-mail tool.

Workflow:
    1. Read every active customer from oc_customer.
    2. For customers with an empty `code`, generate a fresh 40-hex token and
       UPDATE the row. Customers who already have a code (e.g. they hit
       "Forgot Password" in OC already) are left alone — overwriting their
       code would invalidate the link sitting in their inbox.
    3. Build a reset URL per customer:  <store_url>/index.php?route=account/reset&code=<code>
    4. Write CSV: customer_id, email, firstname, lastname, reset_url
       (UTF-8 with BOM so Excel opens Greek names correctly.)
"""
from __future__ import annotations
import csv
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _normalize_store_url(s: str) -> str:
    """Reduce whatever the user pasted to a clean base URL.

    Accepts a bare base (``https://shop.gr``), a path-prefixed base
    (``https://example.com/shop``), or even the full homepage URL
    (``http://localhost/index.php?route=common/home``) and returns the
    base that ``/index.php?route=account/reset&code=…`` should be appended
    to. Without this, naively concatenating a homepage URL produces broken
    links with two ``/index.php?route=`` segments.
    """
    parts = urlsplit(s.strip())
    path = parts.path
    if path.endswith("/index.php"):
        path = path[: -len("/index.php")]
    path = path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


@dataclass
class ResetExportStats:
    total_customers: int
    codes_generated: int
    codes_reused: int
    csv_path: Path


def export_reset_urls(
    engine: Engine,
    prefix: str,
    store_url: str,
    out_path: Path,
    log_cb: Optional[Callable[[str], None]] = None,
) -> ResetExportStats:
    """Populate missing reset codes and write a CSV of reset URLs.

    `store_url` is the public store base (e.g. https://shop.example.gr) —
    a trailing slash is tolerated. The URL written into the CSV always has
    the form `<base>/index.php?route=account/reset&code=<code>`, which works
    on every OC 3.x install regardless of SEF URL settings.
    """
    base = _normalize_store_url(store_url)
    out_path = Path(out_path)

    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    with engine.begin() as conn:
        rows = conn.execute(text(
            f"SELECT customer_id, email, firstname, lastname, code "
            f"FROM `{prefix}customer` WHERE status = 1 "
            f"ORDER BY customer_id"
        )).fetchall()

        generated = 0
        reused = 0
        export_rows: list[tuple[int, str, str, str, str]] = []

        for r in rows:
            code = r.code or ""
            if not code:
                code = secrets.token_hex(20)  # 40 hex chars, matches OC's own format
                conn.execute(text(
                    f"UPDATE `{prefix}customer` SET code = :code "
                    f"WHERE customer_id = :cid"
                ), {"code": code, "cid": r.customer_id})
                generated += 1
            else:
                reused += 1

            url = f"{base}/index.php?route=account/reset&code={code}"
            export_rows.append(
                (r.customer_id, r.email, r.firstname or "", r.lastname or "", url)
            )

    # utf-8-sig writes a BOM so Excel auto-detects UTF-8 (otherwise Greek
    # names render as mojibake when the user opens the file).
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "email", "firstname", "lastname", "reset_url"])
        writer.writerows(export_rows)

    stats = ResetExportStats(
        total_customers=len(export_rows),
        codes_generated=generated,
        codes_reused=reused,
        csv_path=out_path,
    )
    _log(
        f"✓ Wrote {stats.total_customers} reset URLs → {out_path} "
        f"({stats.codes_generated} new codes, {stats.codes_reused} reused)"
    )
    return stats
