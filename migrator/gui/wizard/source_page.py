"""Wizard page 1 — choose source platform and connection."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QLineEdit, QSpinBox, QPushButton, QLabel,
    QFileDialog, QRadioButton, QButtonGroup, QFrame,
)
from PySide6.QtCore import Qt

from migrator.pipeline.importer import DumpImporter
from migrator.pipeline.prefix_detect import (
    detect_prefix, detect_prefix_in_file,
    detect_platform, detect_platform_in_file,
)


PLATFORMS = ["PrestaShop", "Magento 1.x", "Magento 2.x", "WooCommerce"]
DEFAULT_PREFIXES = {
    "PrestaShop": "psxm_",
    "Magento 1.x": "",
    "Magento 2.x": "",
    "WooCommerce": "wp_",
}


class SourcePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)

        title = QLabel("<h2>Source Database</h2>")
        root.addWidget(title)

        # Platform selector
        platform_box = QGroupBox("Source Platform")
        pf = QFormLayout(platform_box)
        self.platform_combo = QComboBox()
        self.platform_combo.addItems(PLATFORMS)
        self.platform_combo.currentTextChanged.connect(self._on_platform_change)
        pf.addRow("Platform:", self.platform_combo)
        prefix_row = QHBoxLayout()
        self.prefix_edit = QLineEdit("psxm_")
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.setToolTip(
            "Auto-detect the source platform and table prefix from the dump file "
            "or database connection."
        )
        self.detect_btn.clicked.connect(self._detect_prefix)
        self.detect_status = QLabel("")
        prefix_row.addWidget(self.prefix_edit, stretch=1)
        prefix_row.addWidget(self.detect_btn)
        prefix_row.addWidget(self.detect_status)
        pf.addRow("Table prefix:", prefix_row)
        root.addWidget(platform_box)

        # Mode selector
        mode_box = QGroupBox("Source Mode")
        mode_layout = QVBoxLayout(mode_box)
        self.radio_file = QRadioButton("Load from SQL dump file (recommended)")
        self.radio_db = QRadioButton("Connect to existing database directly")
        self.radio_file.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.radio_file)
        self._mode_group.addButton(self.radio_db)
        self.radio_file.toggled.connect(self._on_mode_change)
        mode_layout.addWidget(self.radio_file)
        mode_layout.addWidget(self.radio_db)
        root.addWidget(mode_box)

        # SQL file section
        self.file_box = QGroupBox("SQL Dump File")
        ff = QVBoxLayout(self.file_box)
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Path to .sql file…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse_btn)
        ff.addLayout(file_row)

        ff.addWidget(QLabel("<b>Local MySQL/MariaDB to load the dump into:</b>"))
        self.import_form = self._make_db_form("localhost", 3306, "root", "", "migrator_src_tmp")
        ff.addLayout(self.import_form["layout"])
        root.addWidget(self.file_box)

        # Direct DB connection section
        self.db_box = QGroupBox("Database Connection")
        db_layout = QVBoxLayout(self.db_box)
        self.direct_form = self._make_db_form("localhost", 3306, "root", "", "")
        db_layout.addLayout(self.direct_form["layout"])
        self.db_box.hide()
        root.addWidget(self.db_box)

        # Source files root (for image migration)
        self.files_box = QGroupBox("Source Files Root (optional, needed for images)")
        fl = QVBoxLayout(self.files_box)
        fl.addWidget(QLabel(
            "Point to the source platform's installation root directory.\n"
            "For PrestaShop: the folder containing <code>img/p/</code>, <code>img/c/</code>, <code>img/m/</code>.\n"
            "Leave empty to skip image file migration."
        ))
        files_row = QHBoxLayout()
        self.files_edit = QLineEdit()
        self.files_edit.setPlaceholderText("/var/www/prestashop/")
        files_browse = QPushButton("Browse…")
        files_browse.clicked.connect(self._browse_files_root)
        files_row.addWidget(self.files_edit)
        files_row.addWidget(files_browse)
        fl.addLayout(files_row)
        root.addWidget(self.files_box)

        # Test button
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_connection)
        self.test_label = QLabel("")
        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.test_label)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

    def _make_db_form(
        self, host: str, port: int, user: str, password: str, db: str
    ) -> dict:
        form = QFormLayout()
        h = QLineEdit(host)
        p = QSpinBox()
        p.setRange(1, 65535)
        p.setValue(port)
        u = QLineEdit(user)
        pw = QLineEdit(password)
        pw.setEchoMode(QLineEdit.EchoMode.Password)
        d = QLineEdit(db)
        form.addRow("Host:", h)
        form.addRow("Port:", p)
        form.addRow("User:", u)
        form.addRow("Password:", pw)
        form.addRow("Database name:", d)
        return {"layout": form, "host": h, "port": p, "user": u, "password": pw, "db": d}

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SQL Dump", "", "SQL files (*.sql);;All files (*)"
        )
        if path:
            self.file_edit.setText(path)

    def _browse_files_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select source platform root")
        if path:
            self.files_edit.setText(path)

    def _on_platform_change(self, text: str) -> None:
        self.prefix_edit.setText(DEFAULT_PREFIXES.get(text, ""))

    def _on_mode_change(self, file_mode: bool) -> None:
        self.file_box.setVisible(file_mode)
        self.db_box.setVisible(not file_mode)

    def _test_connection(self) -> None:
        cfg = self.get_db_config()
        imp = DumpImporter(cfg["host"], cfg["port"], cfg["user"], cfg["password"])
        ok, msg = imp.test_connection()
        if ok:
            self.test_label.setText("<span style='color:green'>✓ Connected</span>")
        else:
            self.test_label.setText(f"<span style='color:red'>✗ {msg}</span>")

    def _detect_prefix(self) -> None:
        """Auto-detect platform AND prefix together. Updates both the platform combo
        and the prefix field. In file mode scans the .sql file; in direct mode
        queries the DB.
        """
        from migrator.canonical.models import SourcePlatform

        platform_to_label = {
            SourcePlatform.PRESTASHOP:  "PrestaShop",
            SourcePlatform.MAGENTO1:    "Magento 1.x",
            SourcePlatform.MAGENTO2:    "Magento 2.x",
            SourcePlatform.WOOCOMMERCE: "WooCommerce",
        }

        if self.radio_file.isChecked():
            sql_file = self.get_sql_file()
            if not sql_file or str(sql_file) == "." or not sql_file.is_file():
                self.detect_status.setText("<span style='color:red'>pick a .sql file first</span>")
                return
            try:
                result = detect_platform_in_file(sql_file)
            except Exception as e:
                self.detect_status.setText(f"<span style='color:red'>✗ {e.__class__.__name__}</span>")
                return
        else:
            from sqlalchemy import create_engine
            cfg = self.get_db_config()
            if not cfg["db"]:
                self.detect_status.setText("<span style='color:red'>enter DB name first</span>")
                return
            url = (
                f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
                f"@{cfg['host']}:{cfg['port']}/{cfg['db']}?charset=utf8mb4"
            )
            try:
                engine = create_engine(url, pool_pre_ping=True)
                result = detect_platform(engine)
            except Exception as e:
                self.detect_status.setText(f"<span style='color:red'>✗ {e.__class__.__name__}</span>")
                return

        if result is None:
            self.detect_status.setText(
                "<span style='color:red'>✗ no known platform fingerprint found</span>"
            )
            return

        platform, prefix = result
        label = platform_to_label.get(platform)
        if label and self.platform_combo.currentText() != label:
            self.platform_combo.setCurrentText(label)
        self.prefix_edit.setText(prefix)
        shown_prefix = "(empty)" if prefix == "" else prefix
        self.detect_status.setText(
            f"<span style='color:green'>✓ {label}, prefix: {shown_prefix}</span>"
        )

    # ── Public API ────────────────────────────────────────────

    def get_platform(self) -> str:
        return self.platform_combo.currentText()

    def get_prefix(self) -> str:
        return self.prefix_edit.text().strip()

    def is_file_mode(self) -> bool:
        return self.radio_file.isChecked()

    def get_sql_file(self) -> Path:
        return Path(self.file_edit.text().strip())

    def get_db_config(self) -> dict:
        if self.radio_file.isChecked():
            f = self.import_form
        else:
            f = self.direct_form
        return {
            "host": f["host"].text().strip(),
            "port": f["port"].value(),
            "user": f["user"].text().strip(),
            "password": f["password"].text(),
            "db": f["db"].text().strip(),
        }

    def get_image_root(self) -> Path | None:
        text = self.files_edit.text().strip()
        return Path(text) if text else None
