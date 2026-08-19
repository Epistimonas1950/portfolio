"""Main application window — sidebar stepper + stacked wizard pages."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QStackedWidget, QMessageBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont

from migrator.gui.wizard.source_page import SourcePage
from migrator.gui.wizard.target_page import TargetPage
from migrator.gui.wizard.run_page import RunPage
from migrator.gui.workers import ImportWorker, MigrationWorker
from migrator.pipeline.importer import DumpImporter
from migrator.canonical.models import SourcePlatform


STEPS = ["1  Source", "2  Target", "3  Migrate"]

PLATFORM_MAP = {
    "PrestaShop": SourcePlatform.PRESTASHOP,
    "Magento 1.x": SourcePlatform.MAGENTO1,
    "Magento 2.x": SourcePlatform.MAGENTO2,
    "WooCommerce": SourcePlatform.WOOCOMMERCE,
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCart Migrator")
        self.setMinimumSize(900, 640)
        self._import_worker: QThread | None = None
        self._migration_worker: QThread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setFixedWidth(170)
        sidebar.setStyleSheet(
            "QFrame { background: #2c3e50; }"
            "QLabel { color: #ecf0f1; }"
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 20, 0, 20)
        sb_layout.setSpacing(4)

        app_label = QLabel("OpenCart\nMigrator")
        app_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        app_label.setFont(font)
        app_label.setStyleSheet("color: #1abc9c; padding: 10px;")
        sb_layout.addWidget(app_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #34495e;")
        sb_layout.addWidget(sep)
        sb_layout.addSpacing(10)

        self._step_labels: list[QLabel] = []
        for step in STEPS:
            lbl = QLabel(f"  {step}")
            lbl.setMinimumHeight(40)
            lbl.setStyleSheet("color: #7f8c8d; padding: 4px 12px;")
            self._step_labels.append(lbl)
            sb_layout.addWidget(lbl)

        sb_layout.addStretch()

        version_lbl = QLabel("v0.1.1")
        version_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_lbl.setStyleSheet("color: #34495e; font-size: 10px;")
        sb_layout.addWidget(version_lbl)
        outer.addWidget(sidebar)

        # ── Content area ──────────────────────────────────────
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(24, 24, 24, 16)
        content_layout.setSpacing(16)

        self.stack = QStackedWidget()
        self.source_page = SourcePage()
        self.target_page = TargetPage()
        self.run_page = RunPage()

        # Wrap pages in scroll areas for small screens
        for page in (self.source_page, self.target_page, self.run_page):
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.stack.addWidget(scroll)

        content_layout.addWidget(self.stack, stretch=1)

        # Navigation buttons
        nav = QHBoxLayout()
        self.back_btn = QPushButton("← Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setEnabled(False)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._go_next)
        self.next_btn.setMinimumWidth(120)
        nav.addWidget(self.back_btn)
        nav.addStretch()
        nav.addWidget(self.next_btn)
        content_layout.addLayout(nav)

        outer.addWidget(content_frame, stretch=1)

        # Wire up Run button + Run Another (clears log and returns to step 0)
        self.run_page.run_btn.clicked.connect(self._start_migration)
        self.run_page.run_another_btn.clicked.connect(self._reset_for_new_migration)
        self.run_page.export_reset_btn.clicked.connect(self._export_reset_urls)

        self._set_step(0)

    # ── Navigation ────────────────────────────────────────────

    def _set_step(self, index: int) -> None:
        self._current_step = index
        self.stack.setCurrentIndex(index)
        self.back_btn.setEnabled(index > 0)

        if index == len(STEPS) - 1:
            self.next_btn.setVisible(False)
        else:
            self.next_btn.setVisible(True)

        for i, lbl in enumerate(self._step_labels):
            if i == index:
                lbl.setStyleSheet(
                    "color: #1abc9c; font-weight: bold; "
                    "background: #34495e; padding: 4px 12px;"
                )
            elif i < index:
                lbl.setStyleSheet("color: #ecf0f1; padding: 4px 12px;")
            else:
                lbl.setStyleSheet("color: #7f8c8d; padding: 4px 12px;")

    def _go_back(self) -> None:
        if self._current_step > 0:
            self._set_step(self._current_step - 1)

    def _go_next(self) -> None:
        if self._current_step == 0:
            if not self._validate_source():
                return
            if self.source_page.is_file_mode():
                self._start_import()
                return
        elif self._current_step == 1:
            if not self._run_preflight():
                return  # user cancelled or hit a fatal issue
        self._set_step(self._current_step + 1)

    def _run_preflight(self) -> bool:
        from sqlalchemy import create_engine
        from migrator.pipeline.preflight import run_preflight, auto_fix
        from migrator.gui.preflight_dialog import PreflightDialog

        cfg = self.target_page.get_db_config()
        if not cfg["db"]:
            QMessageBox.warning(self, "Missing info", "Please enter the OpenCart database name.")
            return False

        url = (f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
               f"@{cfg['host']}:{cfg['port']}/{cfg['db']}?charset=utf8mb4")
        engine = create_engine(url, pool_pre_ping=True)

        first_check = True
        while True:
            result = run_preflight(engine, cfg["prefix"])
            # Skip the dialog only on the very first check when everything is clean.
            # After an Auto-Fix, always show the result so the user sees the green.
            if first_check and result.is_clean:
                return True
            dialog = PreflightDialog(result, self)
            dialog.exec()
            first_check = False
            if dialog.choice == "fix":
                log = auto_fix(engine, result)
                QMessageBox.information(
                    self, "Auto-Fix Results",
                    "\n".join(log) if log else "Nothing to fix",
                )
                continue
            elif dialog.choice == "continue":
                return True
            else:
                return False

    # ── SQL dump import ───────────────────────────────────────

    def _validate_source(self) -> bool:
        if self.source_page.is_file_mode():
            sql_file = self.source_page.get_sql_file()
            if not sql_file.exists():
                QMessageBox.warning(self, "Missing file", "SQL dump file not found.")
                return False
        cfg = self.source_page.get_db_config()
        if not cfg["db"]:
            QMessageBox.warning(self, "Missing info", "Please enter a database name.")
            return False
        return True

    def _start_import(self) -> None:
        cfg = self.source_page.get_db_config()
        importer = DumpImporter(cfg["host"], cfg["port"], cfg["user"], cfg["password"])
        sql_file = self.source_page.get_sql_file()

        # Switch to run page and show import progress there
        self._set_step(2)
        self.run_page.append_log("── Importing SQL dump ──────────────────")
        self.run_page.set_progress("Importing…", 0, 100)
        self.run_page.set_running(True)
        self.next_btn.setVisible(False)
        self.back_btn.setEnabled(False)

        self._import_worker = ImportWorker(importer, sql_file, cfg["db"])
        self._import_worker.log.connect(self.run_page.append_log)
        self._import_worker.progress.connect(self._on_import_progress)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_worker.start()

    def _on_import_progress(self, sent: int, total: int) -> None:
        mb_sent = sent / 1_048_576
        mb_total = total / 1_048_576
        self.run_page.set_progress(
            f"Importing… {mb_sent:.1f} / {mb_total:.1f} MB",
            sent, total,
        )

    def _on_import_finished(self, success: bool, error: str) -> None:
        self.run_page.set_running(False)
        self.back_btn.setEnabled(True)
        if success:
            self.run_page.append_log("✓ Import complete\n")
            self.run_page.set_progress("Import complete", 1, 1)
            self._set_step(1)  # go to target page; Next runs preflight → step 2 normally
            self.next_btn.setVisible(True)
        else:
            self.run_page.append_log(f"✗ Import failed: {error}")
            QMessageBox.critical(self, "Import Failed", error)

    # ── Migration ─────────────────────────────────────────────

    def _start_migration(self) -> None:
        from sqlalchemy import create_engine
        from migrator.pipeline.runner import MigrationRunner, SourceConfig, TargetConfig, MigrationOptions

        src_cfg = self.source_page.get_db_config()
        tgt_cfg = self.target_page.get_db_config()
        lang_map = self.target_page.get_lang_map()
        img_dir = self.target_page.get_image_dir()
        options_dict = self.run_page.get_options()
        platform_name = self.source_page.get_platform()

        if not tgt_cfg["db"]:
            QMessageBox.warning(self, "Missing info", "Please enter the OpenCart database name on the Target page.")
            return

        def make_url(cfg: dict) -> str:
            return (
                f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
                f"@{cfg['host']}:{cfg['port']}/{cfg['db']}?charset=utf8mb4"
            )

        src_engine = create_engine(make_url(src_cfg), pool_pre_ping=True)
        tgt_engine = create_engine(make_url(tgt_cfg), pool_pre_ping=True)

        source = SourceConfig(
            platform=PLATFORM_MAP[platform_name],
            engine=src_engine,
            table_prefix=self.source_page.get_prefix(),
            lang_map=lang_map,
            image_root=self.source_page.get_image_root(),
        )
        target = TargetConfig(
            engine=tgt_engine,
            table_prefix=tgt_cfg["prefix"],
            oc_image_dir=img_dir,
            store_id=0,
        )
        options = MigrationOptions(
            categories=options_dict["categories"],
            manufacturers=options_dict["manufacturers"],
            products=options_dict["products"],
            customers=options_dict["customers"],
            orders=options_dict["orders"],
        )

        if self.run_page.is_wipe():
            from migrator.pipeline.wipe import wipe_target, WIPE_TABLES
            confirm = QMessageBox.warning(
                self, "Confirm wipe",
                f"This will TRUNCATE these tables in `{tgt_cfg['db']}`:\n\n"
                + ", ".join(WIPE_TABLES) + "\n\n"
                "Demo products/categories/customers will be deleted too.\n"
                "OpenCart config (settings, languages, countries, currencies) is preserved.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            self.run_page.append_log("── Wiping target tables ──────────────")
            wipe_target(tgt_engine, tgt_cfg["prefix"], log_cb=self.run_page.append_log)
            self.run_page.append_log("✓ Wipe complete\n")

        runner = MigrationRunner(source, target, options)

        self.run_page.append_log("── Starting migration ──────────────────")
        self.run_page.set_running(True)

        self._migration_worker = MigrationWorker(runner)
        self._migration_worker.log.connect(self.run_page.append_log)
        self._migration_worker.progress.connect(self.run_page.set_progress)
        self._migration_worker.finished.connect(self._on_migration_finished)
        self._migration_worker.start()

    def _on_migration_finished(self, success: bool, error: str, stats: dict) -> None:
        self.run_page.set_running(False)
        if success:
            summary = "  ".join(f"{k}: {v}" for k, v in stats.items())
            self.run_page.append_log(f"\n✓ Migration complete — {summary}")
            self.run_page.set_progress("Done", 1, 1)
            # Reset-URL export only makes sense once we've successfully populated
            # oc_customer; offer it as a follow-up action.
            self.run_page.show_export_reset()
        else:
            self.run_page.append_log(f"\n✗ Migration failed: {error}")
            QMessageBox.critical(self, "Migration Failed", error)
        # Either way, offer to start over
        self.run_page.show_run_another()

    def _export_reset_urls(self) -> None:
        from migrator.gui.reset_urls_dialog import ResetUrlsDialog
        from migrator.pipeline.reset_urls import export_reset_urls
        from sqlalchemy import create_engine

        tgt_cfg = self.target_page.get_db_config()
        if not tgt_cfg["db"]:
            QMessageBox.warning(self, "Target not configured",
                                "The target database connection is missing.")
            return

        dlg = ResetUrlsDialog(self)
        if dlg.exec() != ResetUrlsDialog.DialogCode.Accepted:
            return
        store_url, out_path = dlg.get_values()

        try:
            url = (
                f"mysql+pymysql://{tgt_cfg['user']}:{tgt_cfg['password']}"
                f"@{tgt_cfg['host']}:{tgt_cfg['port']}/{tgt_cfg['db']}?charset=utf8mb4"
            )
            engine = create_engine(url, pool_pre_ping=True)
            self.run_page.append_log("\n── Exporting password-reset URLs ─────")
            stats = export_reset_urls(
                engine, tgt_cfg["prefix"], store_url, out_path,
                log_cb=self.run_page.append_log,
            )
            QMessageBox.information(
                self, "Export complete",
                f"Wrote {stats.total_customers} customers to:\n{stats.csv_path}\n\n"
                f"{stats.codes_generated} new reset codes generated, "
                f"{stats.codes_reused} existing codes reused.",
            )
        except Exception as e:
            self.run_page.append_log(f"✗ Export failed: {e}")
            QMessageBox.critical(self, "Export failed", str(e))

    def _reset_for_new_migration(self) -> None:
        """Return to step 0 with a fresh log/progress so the user can do another migration."""
        self.run_page.reset_for_new_run()
        self._set_step(0)
