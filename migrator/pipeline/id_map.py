"""Sidecar table that persists source_id → oc_id mappings across runs.

Lives in the OpenCart target database next to the OC tables so the user can
see/edit/diagnose it with the same tools. One row per (platform, entity, source_id);
writers consult it before inserting and skip any row whose source_id is already
mapped. The runner clears the table when the user opts in to wipe-before-run.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


# Entity-type constants — keep these stable; they're persisted in the table.
ENTITY_CATEGORY = "category"
ENTITY_MANUFACTURER = "manufacturer"
ENTITY_PRODUCT = "product"
ENTITY_CUSTOMER = "customer"
ENTITY_ORDER = "order"


_TABLE_NAME = "migrator_id_map"


def _ddl(prefix: str) -> str:
    # InnoDB so the UPSERT semantics + crash recovery match the target schema's
    # newer tables (writers already require InnoDB after the preflight fix).
    # Pin COLLATE so a dump from a MySQL 8 source imports cleanly into a
    # MariaDB target — MariaDB rejects MySQL 8's default utf8mb4_0900_ai_ci.
    return (
        f"CREATE TABLE IF NOT EXISTS `{prefix}{_TABLE_NAME}` ("
        f"  source_platform VARCHAR(32) NOT NULL,"
        f"  entity_type     VARCHAR(32) NOT NULL,"
        f"  source_id       BIGINT NOT NULL,"
        f"  oc_id           INT UNSIGNED NOT NULL,"
        f"  created_at      DATETIME NOT NULL,"
        f"  PRIMARY KEY (source_platform, entity_type, source_id),"
        f"  KEY idx_lookup (source_platform, entity_type)"
        f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    )


class IdMapStore:
    """Per-platform, per-entity source_id → oc_id persistence.

    Threading model: a single IdMapStore instance is shared across all writers
    in a run. Each writer calls `.load(entity_type)` once at start to pull the
    existing mappings, then calls `.record(conn, entity_type, source_id, oc_id)`
    inside its own transaction after each successful insert so the mapping is
    committed atomically with the row it points at.
    """

    def __init__(self, engine: Engine, prefix: str, source_platform: str) -> None:
        self.engine = engine
        self.prefix = prefix
        self.platform = source_platform

    @property
    def table(self) -> str:
        return f"`{self.prefix}{_TABLE_NAME}`"

    def ensure_table(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(_ddl(self.prefix)))
            # Heal tables created before the COLLATE pin landed — otherwise a
            # dump off a MySQL 8 source still carries utf8mb4_0900_ai_ci and
            # fails to import on MariaDB targets.
            row = conn.execute(text(
                "SELECT TABLE_COLLATION FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :name"
            ), {"name": f"{self.prefix}{_TABLE_NAME}"}).fetchone()
            if row and row[0] != "utf8mb4_unicode_ci":
                conn.execute(text(
                    f"ALTER TABLE {self.table} "
                    f"CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                ))

    def load(self, entity_type: str) -> dict[int, int]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT source_id, oc_id FROM {self.table} "
                f"WHERE source_platform = :p AND entity_type = :e"
            ), {"p": self.platform, "e": entity_type}).fetchall()
        return {int(r.source_id): int(r.oc_id) for r in rows}

    def record(
        self,
        conn: Connection,
        entity_type: str,
        source_id: int,
        oc_id: int,
    ) -> None:
        """Record one mapping. Must be called inside a writer transaction so
        the mapping commits atomically with the row it describes.

        INSERT IGNORE so a duplicate write (e.g. a writer that records before
        checking) is a no-op rather than a constraint error — the existing
        mapping is the truth.
        """
        conn.execute(text(
            f"INSERT IGNORE INTO {self.table} "
            f"(source_platform, entity_type, source_id, oc_id, created_at) "
            f"VALUES (:p, :e, :sid, :ocid, NOW())"
        ), {"p": self.platform, "e": entity_type, "sid": source_id, "ocid": oc_id})

    def wipe_all(self) -> None:
        """TRUNCATE the entire id_map table. Called from wipe_target() so a
        wipe leaves the sidecar consistent with the empty OC tables.
        """
        with self.engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.table}"))


def chunked(items: list, batch_size: int) -> Iterator[list]:
    """Yield successive `batch_size`-sized slices of `items`. Used by writers
    to wrap inserts in per-batch transactions, so a crash during a long run
    only loses up to `batch_size` rows of progress.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# Convenient default — small enough that a crash doesn't lose much work, big
# enough that the per-transaction overhead is negligible on tables with 100k+ rows.
DEFAULT_BATCH_SIZE = 500
