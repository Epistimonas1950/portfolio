"""Preflight checks for OpenCart 3.x target database before migration."""
from __future__ import annotations
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Tables the migrator writes to (without prefix)
TARGET_TABLES = [
    "category", "category_description", "category_path", "category_to_store",
    "manufacturer", "manufacturer_to_store",
    "product", "product_description", "product_image",
    "product_to_store", "product_to_category",
    "customer", "address",
    "order", "order_product", "order_total", "order_history",
    "country", "zone", "currency", "store",
]


@dataclass
class PreflightResult:
    connection_ok: bool = False
    error: str = ""
    missing_tables: list[str] = field(default_factory=list)
    myisam_tables: list[str] = field(default_factory=list)
    utf8mb3_tables: list[str] = field(default_factory=list)
    language_count: int = 0

    @property
    def is_fatal(self) -> bool:
        """Issues that prevent migration entirely (must cancel)."""
        return bool(self.error or self.missing_tables) or self.language_count == 0

    @property
    def is_fixable(self) -> bool:
        """Issues we can auto-fix with ALTER TABLE."""
        return bool(self.myisam_tables or self.utf8mb3_tables)

    @property
    def is_clean(self) -> bool:
        return not self.is_fatal and not self.is_fixable


def run_preflight(engine: Engine, prefix: str) -> PreflightResult:
    result = PreflightResult()
    expected = {f"{prefix}{t}" for t in TARGET_TABLES}

    try:
        with engine.connect() as conn:
            result.connection_ok = True

            rows = conn.execute(text(
                "SELECT TABLE_NAME, ENGINE, TABLE_COLLATION "
                "FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE()"
            )).fetchall()

            found = {r.TABLE_NAME for r in rows if r.TABLE_NAME in expected}
            result.missing_tables = sorted(expected - found)

            for r in rows:
                if r.TABLE_NAME in expected:
                    if r.ENGINE and r.ENGINE.upper() != "INNODB":
                        result.myisam_tables.append(r.TABLE_NAME)
                    if "utf8mb3" in (r.TABLE_COLLATION or "").lower():
                        result.utf8mb3_tables.append(r.TABLE_NAME)

            try:
                row = conn.execute(text(f"SELECT COUNT(*) AS n FROM `{prefix}language`")).fetchone()
                result.language_count = row.n if row else 0
            except Exception:
                pass

    except Exception as e:
        result.error = str(e)

    return result


def auto_fix(engine: Engine, result: PreflightResult) -> list[str]:
    """Run ALTER TABLE on each affected table. Returns log messages."""
    log: list[str] = []
    if not result.is_fixable:
        return log

    with engine.begin() as conn:
        conn.execute(text("SET SESSION sql_mode = ''"))

        for tbl in result.myisam_tables:
            try:
                conn.execute(text(f"ALTER TABLE `{tbl}` ENGINE=InnoDB"))
                log.append(f"✓ {tbl} → InnoDB")
            except Exception as e:
                log.append(f"✗ {tbl} engine change failed: {e}")

        for tbl in result.utf8mb3_tables:
            try:
                conn.execute(text(
                    f"ALTER TABLE `{tbl}` CONVERT TO CHARACTER SET utf8mb4 "
                    f"COLLATE utf8mb4_unicode_ci"
                ))
                log.append(f"✓ {tbl} → utf8mb4")
            except Exception as e:
                log.append(f"✗ {tbl} charset change failed: {e}")

    return log
