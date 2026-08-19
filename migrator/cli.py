"""Command-line interface for the OpenCart Migrator.

Subcommands:
  migrate          — run the full source→target migration
  import-dump      — drop, recreate, and load a SQL dump
  detect-platform  — auto-detect the source platform AND prefix
  detect-prefix    — detect a specific platform's table prefix
  preflight        — check the OpenCart target DB for known issues
  wipe             — TRUNCATE the OC migration tables

Examples:
  python3 -m migrator.cli detect-platform --file dump.sql
  python3 -m migrator.cli detect-prefix --platform woocommerce --file dump.sql
  python3 -m migrator.cli import-dump --file dump.sql --db migrator_src_tmp \\
      --user migrator --password migrator_pass
  python3 -m migrator.cli migrate \\
      --source-platform woocommerce --source-db migrator_src_tmp \\
      --source-prefix Ldi8kDAX_ --source-user migrator --source-password migrator_pass \\
      --target-db opencart --target-prefix oc_ \\
      --target-user migrator --target-password migrator_pass \\
      --entities all --wipe-first
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

from migrator.canonical.models import SourcePlatform

PLATFORM_CHOICES = {
    "prestashop":  SourcePlatform.PRESTASHOP,
    "magento1":    SourcePlatform.MAGENTO1,
    "magento2":    SourcePlatform.MAGENTO2,
    "woocommerce": SourcePlatform.WOOCOMMERCE,
}

DEFAULT_LANG_MAPS = {
    "prestashop":  "2:1",   # Greek → OC English slot
    "magento1":    "0:1",   # admin store → OC slot 1
    "magento2":    "0:1",   # admin store → OC slot 1
    "woocommerce": "1:1",   # single-language WC
}


def _err(msg: str) -> None:
    sys.stderr.write(f"✗ {msg}\n")


def _make_engine(host: str, port: int, user: str, password: str, db: str):
    from sqlalchemy import create_engine
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def _parse_lang_map(s: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        try:
            src, dst = pair.split(":", 1)
            out[int(src)] = int(dst)
        except (ValueError, IndexError):
            raise SystemExit(f"Invalid --lang-map entry: {pair!r}. Use src:dst[,src:dst].")
    return out


# ── Subcommands ───────────────────────────────────────────────

def cmd_detect_prefix(args: argparse.Namespace) -> int:
    from migrator.pipeline.prefix_detect import detect_prefix, detect_prefix_in_file
    platform = PLATFORM_CHOICES[args.platform]

    if args.file:
        sql_file = Path(args.file)
        if not sql_file.is_file():
            _err(f"File not found: {sql_file}")
            return 1
        prefix = detect_prefix_in_file(sql_file, platform)
    else:
        if not args.db:
            _err("Either --file or --db must be specified.")
            return 1
        engine = _make_engine(args.host, args.port, args.user, args.password, args.db)
        prefix = detect_prefix(engine, platform)

    if prefix is None:
        _err("No matching tables found — is this the right platform?")
        return 1
    print(prefix if prefix else "(empty prefix)")
    return 0


def cmd_detect_platform(args: argparse.Namespace) -> int:
    """Auto-detect the source platform (and its prefix) from a dump or DB."""
    from migrator.pipeline.prefix_detect import detect_platform, detect_platform_in_file

    if args.file:
        sql_file = Path(args.file)
        if not sql_file.is_file():
            _err(f"File not found: {sql_file}")
            return 1
        result = detect_platform_in_file(sql_file)
    else:
        if not args.db:
            _err("Either --file or --db must be specified.")
            return 1
        engine = _make_engine(args.host, args.port, args.user, args.password, args.db)
        result = detect_platform(engine)

    if result is None:
        _err("No known platform fingerprint found.")
        return 1
    platform, prefix = result
    print(f"platform={platform.value}")
    print(f"prefix={prefix if prefix else '(empty)'}")
    return 0


def cmd_import_dump(args: argparse.Namespace) -> int:
    from migrator.pipeline.importer import DumpImporter, ImportError as DumpImportError
    sql_file = Path(args.file)
    if not sql_file.is_file():
        _err(f"File not found: {sql_file}")
        return 1
    imp = DumpImporter(args.host, args.port, args.user, args.password)
    print(f"→ Dropping database '{args.db}' (if any)…")
    imp.drop_database(args.db)
    print(f"→ Creating database '{args.db}'…")
    try:
        imp.create_database(args.db)
    except DumpImportError as e:
        _err(str(e))
        return 1
    size_mb = sql_file.stat().st_size / 1_048_576
    print(f"→ Loading {sql_file.name} ({size_mb:.1f} MB)…")

    last_pct = [-1]
    def progress(sent: int, total: int) -> None:
        pct = int(100 * sent / total) if total else 0
        if pct >= last_pct[0] + 5:
            print(f"  … {pct}% ({sent/1_048_576:.1f}/{total/1_048_576:.1f} MB)")
            last_pct[0] = pct

    try:
        imp.load_dump(sql_file, args.db, progress)
    except DumpImportError as e:
        _err(str(e))
        return 1
    print("✓ Import complete")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from migrator.pipeline.preflight import run_preflight, auto_fix
    engine = _make_engine(args.host, args.port, args.user, args.password, args.db)
    result = run_preflight(engine, args.prefix)

    if result.error:
        _err(f"Connection error: {result.error}")
        return 1

    print(f"Connection: OK")
    print(f"Languages: {result.language_count}")
    print(f"Missing tables: {len(result.missing_tables)}")
    if result.missing_tables:
        print("  " + ", ".join(result.missing_tables))
    print(f"MyISAM tables: {len(result.myisam_tables)}")
    if result.myisam_tables:
        print("  " + ", ".join(result.myisam_tables))
    print(f"utf8mb3 tables: {len(result.utf8mb3_tables)}")
    if result.utf8mb3_tables:
        print("  " + ", ".join(result.utf8mb3_tables))

    if result.is_clean:
        print("✓ Target is clean")
        return 0
    if result.is_fatal:
        _err("Fatal issues found — cannot proceed.")
        return 1
    if result.is_fixable and args.auto_fix:
        print("→ Running auto-fix…")
        log = auto_fix(engine, result)
        for line in log:
            print(f"  {line}")
        print("✓ Auto-fix complete")
        return 0
    if result.is_fixable:
        _err("Fixable issues found. Re-run with --auto-fix to apply ALTER TABLE statements.")
        return 1
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    if not args.confirm:
        _err("Add --confirm to actually wipe (destructive operation).")
        return 1
    from migrator.pipeline.wipe import wipe_target
    engine = _make_engine(args.host, args.port, args.user, args.password, args.db)
    print(f"→ Wiping migration tables in '{args.db}'…")
    wipe_target(engine, args.prefix, log_cb=lambda m: print(m))
    print("✓ Wipe complete")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from migrator.pipeline.runner import (
        MigrationRunner, SourceConfig, TargetConfig, MigrationOptions,
    )

    platform = PLATFORM_CHOICES[args.source_platform]

    lang_map_raw = args.lang_map or DEFAULT_LANG_MAPS[args.source_platform]
    lang_map = _parse_lang_map(lang_map_raw)

    entities = args.entities.lower().split(",")
    if "all" in entities:
        opts = MigrationOptions(
            categories=True, manufacturers=True,
            products=True, customers=True, orders=True,
        )
    else:
        opts = MigrationOptions(
            categories="categories" in entities or "cat" in entities,
            manufacturers="manufacturers" in entities or "manu" in entities,
            products="products" in entities or "prod" in entities,
            customers="customers" in entities or "cust" in entities,
            orders="orders" in entities,
        )

    src_engine = _make_engine(
        args.source_host, args.source_port, args.source_user,
        args.source_password, args.source_db,
    )
    tgt_engine = _make_engine(
        args.target_host, args.target_port, args.target_user,
        args.target_password, args.target_db,
    )

    if args.wipe_first:
        from migrator.pipeline.wipe import wipe_target
        print("→ Wiping target migration tables…")
        wipe_target(tgt_engine, args.target_prefix, log_cb=lambda m: print(m))

    source = SourceConfig(
        platform=platform,
        engine=src_engine,
        table_prefix=args.source_prefix,
        lang_map=lang_map,
        image_root=Path(args.source_image_root) if args.source_image_root else None,
    )
    target = TargetConfig(
        engine=tgt_engine,
        table_prefix=args.target_prefix,
        oc_image_dir=Path(args.target_image_dir) if args.target_image_dir else None,
        store_id=args.target_store_id,
    )

    runner = MigrationRunner(
        source, target, opts,
        log_cb=lambda m: print(m),
        progress_cb=None,  # CLI doesn't need per-row progress — log_cb already throttles
    )
    try:
        stats = runner.run()
    except Exception as e:
        _err(f"Migration failed: {e}")
        return 1

    print("\n✓ Migration complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


# ── Argument parser ───────────────────────────────────────────

def _add_db_args(p: argparse.ArgumentParser, role: str = "") -> None:
    """Add --[role-]host/--[role-]port/etc. flags."""
    prefix = f"{role}-" if role else ""
    p.add_argument(f"--{prefix}host", default="localhost")
    p.add_argument(f"--{prefix}port", type=int, default=3306)
    p.add_argument(f"--{prefix}user", default="root")
    p.add_argument(f"--{prefix}password", default="")
    p.add_argument(f"--{prefix}db", required=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrator",
        description="OpenCart 3.x migration tool (CLI)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # detect-prefix
    p_dp = sub.add_parser("detect-prefix", help="Auto-detect a source platform's table prefix.")
    p_dp.add_argument("--platform", required=True, choices=list(PLATFORM_CHOICES))
    p_dp.add_argument("--file", help="SQL dump file (use this OR DB connection flags)")
    _add_db_args(p_dp)
    p_dp.set_defaults(func=cmd_detect_prefix)

    # detect-platform
    p_dpl = sub.add_parser(
        "detect-platform",
        help="Auto-detect the source platform AND its table prefix (no --platform required).",
    )
    p_dpl.add_argument("--file", help="SQL dump file (use this OR DB connection flags)")
    _add_db_args(p_dpl)
    p_dpl.set_defaults(func=cmd_detect_platform)

    # import-dump
    p_im = sub.add_parser("import-dump", help="Drop, recreate, and load a SQL dump.")
    p_im.add_argument("--file", required=True)
    _add_db_args(p_im)
    p_im.set_defaults(func=cmd_import_dump)

    # preflight
    p_pf = sub.add_parser("preflight", help="Check OC target DB for known issues.")
    _add_db_args(p_pf)
    p_pf.add_argument("--prefix", default="oc_")
    p_pf.add_argument("--auto-fix", action="store_true",
                      help="Apply ALTER TABLE fixes for MyISAM/utf8mb3 tables.")
    p_pf.set_defaults(func=cmd_preflight)

    # wipe
    p_w = sub.add_parser("wipe", help="TRUNCATE the OC migration tables.")
    _add_db_args(p_w)
    p_w.add_argument("--prefix", default="oc_")
    p_w.add_argument("--confirm", action="store_true",
                     help="Required — confirms the destructive operation.")
    p_w.set_defaults(func=cmd_wipe)

    # migrate
    p_m = sub.add_parser("migrate", help="Run a source→OC migration.")
    p_m.add_argument("--source-platform", required=True, choices=list(PLATFORM_CHOICES))
    _add_db_args(p_m, role="source")
    p_m.add_argument("--source-prefix", default="")
    p_m.add_argument("--source-image-root", help="Path to source platform root (for image migration).")
    _add_db_args(p_m, role="target")
    p_m.add_argument("--target-prefix", default="oc_")
    p_m.add_argument("--target-image-dir", help="Path to OpenCart image/ directory.")
    p_m.add_argument("--target-store-id", type=int, default=0)
    p_m.add_argument("--lang-map", default=None,
                     help="Language map, e.g. '2:1,1:1'. Defaults per platform.")
    p_m.add_argument("--entities", default="all",
                     help="Comma-separated: categories,manufacturers,products,customers,orders or 'all'.")
    p_m.add_argument("--wipe-first", action="store_true",
                     help="TRUNCATE OC migration tables before running.")
    p_m.set_defaults(func=cmd_migrate)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate required --*-db for the migrate subcommand
    if args.cmd == "migrate":
        if not args.source_db or not args.target_db:
            parser.error("--source-db and --target-db are required for 'migrate'.")
    elif args.cmd in ("import-dump", "preflight", "wipe"):
        if not args.db:
            parser.error("--db is required.")

    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
