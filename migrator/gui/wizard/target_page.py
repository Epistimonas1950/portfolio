"""Wizard page 2 — OpenCart 3.x target configuration."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QPushButton, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt

from migrator.pipeline.importer import DumpImporter


class TargetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)

        root.addWidget(QLabel("<h2>OpenCart 3.x Target</h2>"))

        # DB connection
        db_box = QGroupBox("Database Connection")
        df = QFormLayout(db_box)
        self.host = QLineEdit("localhost")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(3306)
        self.user = QLineEdit("root")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_name = QLineEdit()
        self.db_name.setPlaceholderText("opencart database name")
        self.prefix = QLineEdit("oc_")
        df.addRow("Host:", self.host)
        df.addRow("Port:", self.port)
        df.addRow("User:", self.user)
        df.addRow("Password:", self.password)
        df.addRow("Database:", self.db_name)
        df.addRow("Table prefix:", self.prefix)

        test_row = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test_connection)
        self.test_label = QLabel("")
        test_row.addWidget(test_btn)
        test_row.addWidget(self.test_label)
        test_row.addStretch()
        df.addRow("", test_row)
        root.addWidget(db_box)

        # Images directory
        img_box = QGroupBox("OpenCart Images Directory (optional)")
        il = QVBoxLayout(img_box)
        il.addWidget(QLabel(
            "Point to your OpenCart installation's <code>image/</code> folder.\n"
            "Leave empty to skip image migration."
        ))
        img_row = QHBoxLayout()
        self.img_dir = QLineEdit()
        self.img_dir.setPlaceholderText("/var/www/opencart/image/")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_img_dir)
        img_row.addWidget(self.img_dir)
        img_row.addWidget(browse_btn)
        il.addLayout(img_row)
        root.addWidget(img_box)

        # Language mapping
        lang_box = QGroupBox("Language Mapping  (source lang ID → OpenCart lang ID)")
        ll = QVBoxLayout(lang_box)
        ll.addWidget(QLabel(
            "Map source platform language IDs to OpenCart language IDs.\n"
            "PrestaShop: Greek=2, English=1. Magento 1.x/2.x: use store_id (admin=0, default=1).\n"
            "OpenCart default: English=1."
        ))
        self.lang_table = QTableWidget(2, 2)
        self.lang_table.setHorizontalHeaderLabels(["Source Language ID", "OpenCart Language ID"])
        self.lang_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.lang_table.setMaximumHeight(120)
        # Default rows: Greek(2)→OC(1) and English(1)→OC(1)
        defaults = [("2", "1"), ("1", "1")]
        for row, (src, tgt) in enumerate(defaults):
            self.lang_table.setItem(row, 0, QTableWidgetItem(src))
            self.lang_table.setItem(row, 1, QTableWidgetItem(tgt))

        add_row_btn = QPushButton("+ Add row")
        add_row_btn.clicked.connect(self._add_lang_row)
        ll.addWidget(self.lang_table)
        ll.addWidget(add_row_btn)
        root.addWidget(lang_box)

        root.addStretch()

    def _test_connection(self) -> None:
        cfg = self.get_db_config()
        imp = DumpImporter(cfg["host"], cfg["port"], cfg["user"], cfg["password"])
        ok, msg = imp.test_connection()
        if ok:
            self.test_label.setText("<span style='color:green'>✓ Connected</span>")
        else:
            self.test_label.setText(f"<span style='color:red'>✗ {msg}</span>")

    def _browse_img_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select OpenCart image/ directory")
        if path:
            self.img_dir.setText(path)

    def _add_lang_row(self) -> None:
        self.lang_table.insertRow(self.lang_table.rowCount())

    # ── Public API ────────────────────────────────────────────

    def get_db_config(self) -> dict:
        return {
            "host": self.host.text().strip(),
            "port": self.port.value(),
            "user": self.user.text().strip(),
            "password": self.password.text(),
            "db": self.db_name.text().strip(),
            "prefix": self.prefix.text().strip(),
        }

    def get_image_dir(self) -> Path | None:
        text = self.img_dir.text().strip()
        return Path(text) if text else None

    def get_lang_map(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for row in range(self.lang_table.rowCount()):
            src_item = self.lang_table.item(row, 0)
            tgt_item = self.lang_table.item(row, 1)
            if src_item and tgt_item:
                try:
                    result[int(src_item.text())] = int(tgt_item.text())
                except ValueError:
                    pass
        return result
