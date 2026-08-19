"""Detect the table prefix used by a source database or SQL dump file.

Each supported platform has a fingerprint table that's unique to it. We look
for any table whose name ends with the fingerprint, then strip the suffix to
recover whatever prefix the install uses.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from migrator.canonical.models import SourcePlatform

# Fingerprint table names (without prefix) per platform. Each must be:
#  - unique to that platform (so other CMS dumps in the same DB don't confuse us)
#  - a table every install of that platform has
PLATFORM_FINGERPRINTS: dict[SourcePlatform, str] = {
    SourcePlatform.PRESTASHOP:  "configuration",            # ps_configuration
    SourcePlatform.MAGENTO1:    "core_resource",            # core_resource (M1-only)
    SourcePlatform.MAGENTO2:    "setup_module",             # setup_module (M2-only; replaces core_resource)
    SourcePlatform.WOOCOMMERCE: "woocommerce_order_items",  # wp_woocommerce_order_items
}


def detect_prefix(engine: Engine, platform: SourcePlatform) -> Optional[str]:
    """Return the detected prefix string (may be ''), or None if not found."""
    fingerprint = PLATFORM_FINGERPRINTS.get(platform)
    if not fingerprint:
        return None

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE :pat"
        ), {"pat": f"%{fingerprint}"}).fetchall()

    # Prefer the shortest matching prefix (most installs use one prefix)
    candidates = []
    for r in rows:
        name = r.TABLE_NAME
        if name == fingerprint:
            candidates.append("")
        elif name.endswith("_" + fingerprint):
            candidates.append(name[:-len(fingerprint)])
        elif name.endswith(fingerprint):
            # match without an underscore separator (rare, but allow it)
            candidates.append(name[:-len(fingerprint)])

    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def detect_platform(engine: Engine) -> Optional[tuple[SourcePlatform, str]]:
    """Probe the connected DB and return (platform, prefix) for the first matching
    fingerprint, or None if nothing matches.

    Platforms with more specific fingerprints (MAGENTO2, WOOCOMMERCE) are checked
    before more general ones (PRESTASHOP) so that, e.g., a WooCommerce install
    doesn't also match anything generic.
    """
    with engine.connect() as conn:
        names = {
            r.TABLE_NAME
            for r in conn.execute(text(
                "SELECT TABLE_NAME FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE()"
            )).fetchall()
        }
    return _match_platform_from_names(names)


def detect_platform_in_file(sql_file: Path) -> Optional[tuple[SourcePlatform, str]]:
    """Stream-scan a SQL dump for CREATE TABLE statements and return
    (platform, prefix) for the first matching fingerprint, or None.
    """
    create_pat = re.compile(
        rb"^\s*CREATE\s+TABLE\s+`([^`]+)`",
        re.IGNORECASE,
    )
    names: set[str] = set()
    try:
        with open(sql_file, "rb") as f:
            for line in f:
                m = create_pat.match(line)
                if not m:
                    continue
                names.add(m.group(1).decode("utf-8", errors="replace"))
    except OSError:
        return None
    return _match_platform_from_names(names)


# Order matters: check the more specific fingerprints first so that, e.g., a
# WordPress install with WooCommerce isn't mistaken for plain WP, and Magento 2
# isn't mistaken for Magento 1.
_PLATFORM_PROBE_ORDER = [
    SourcePlatform.MAGENTO2,
    SourcePlatform.MAGENTO1,
    SourcePlatform.WOOCOMMERCE,
    SourcePlatform.PRESTASHOP,
]


def _match_platform_from_names(names: set[str]) -> Optional[tuple[SourcePlatform, str]]:
    for platform in _PLATFORM_PROBE_ORDER:
        fingerprint = PLATFORM_FINGERPRINTS[platform]
        # Collect every table name ending with the fingerprint, then pick the
        # shortest prefix — same dedupe logic detect_prefix uses to skip plugin
        # tables like `psxm_codfee_configuration`.
        candidates: list[str] = []
        for name in names:
            if name == fingerprint:
                candidates.append("")
            elif name.endswith("_" + fingerprint):
                candidates.append(name[:-len(fingerprint)])
            elif name.endswith(fingerprint) and name != fingerprint:
                candidates.append(name[:-len(fingerprint)])
        if not candidates:
            continue
        candidates.sort(key=len)
        return platform, candidates[0]
    return None


def detect_prefix_in_file(sql_file: Path, platform: SourcePlatform) -> Optional[str]:
    """Stream-scan a SQL dump for a CREATE TABLE matching the platform fingerprint.

    Returns the prefix (may be ''), or None if no fingerprint table is found.
    """
    fingerprint = PLATFORM_FINGERPRINTS.get(platform)
    if not fingerprint:
        return None

    # Matches:  CREATE TABLE `<prefix>fingerprint`
    # Captures the prefix (which may be empty).
    pattern = re.compile(
        rb"^\s*CREATE\s+TABLE\s+`([^`]*?)" + re.escape(fingerprint.encode()) + rb"`",
        re.IGNORECASE,
    )

    candidates: list[str] = []
    try:
        with open(sql_file, "rb") as f:
            for line in f:
                m = pattern.match(line)
                if not m:
                    continue
                candidate = m.group(1).decode("utf-8", errors="replace")
                # Must end with '_' separator, or be empty (unprefixed install)
                if candidate == "" or candidate.endswith("_"):
                    candidates.append(candidate)
    except OSError:
        return None

    if not candidates:
        return None
    # Prefer the shortest prefix — module/plugin tables like `psxm_codfee_configuration`
    # share the fingerprint name but have a longer prefix than the real platform table.
    candidates.sort(key=len)
    return candidates[0]
