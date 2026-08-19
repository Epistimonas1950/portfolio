"""Wizard page 3 — entity selection, progress, and live log."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QPushButton, QLabel, QProgressBar, QTextEdit, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QFont


class RunPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)

        root.addWidget(QLabel("<h2>Run Migration</h2>"))

        # Entity selection
        entity_box = QGroupBox("What to Migrate")
        el = QHBoxLayout(entity_box)
        self.cb_categories = QCheckBox("Categories")
        self.cb_manufacturers = QCheckBox("Manufacturers")
        self.cb_products = QCheckBox("Products")
        self.cb_customers = QCheckBox("Customers")
        self.cb_orders = QCheckBox("Orders")
        self.cb_categories.setChecked(True)
        el.addWidget(self.cb_categories)
        el.addWidget(self.cb_manufacturers)
        el.addWidget(self.cb_products)
        el.addWidget(self.cb_customers)
        el.addWidget(self.cb_orders)
        el.addStretch()
        root.addWidget(entity_box)

        # Wipe-before-run option
        self.cb_wipe = QCheckBox("Wipe target migration tables before starting (destructive)")
        self.cb_wipe.setStyleSheet("QCheckBox { color: #c0392b; }")
        self.cb_wipe.setChecked(False)
        root.addWidget(self.cb_wipe)

        # Progress
        progress_box = QGroupBox("Progress")
        pl = QVBoxLayout(progress_box)
        self.progress_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pl.addWidget(self.progress_label)
        pl.addWidget(self.progress_bar)
        root.addWidget(progress_box)

        # Log
        root.addWidget(QLabel("<b>Migration Log</b>"))
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setUndoRedoEnabled(False)
        self.log_area.document().setMaximumBlockCount(2000)
        font = QFont("Monospace", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.log_area.setFont(font)
        root.addWidget(self.log_area, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Start Migration")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self.log_area.clear)
        self.run_another_btn = QPushButton("↻  Run Another Migration")
        self.run_another_btn.setMinimumHeight(36)
        self.run_another_btn.setVisible(False)
        # Shown only after a successful migration — generates a CSV of
        # password-reset URLs for the bulk-mail customer invite.
        self.export_reset_btn = QPushButton("✉  Export Password-Reset CSV…")
        self.export_reset_btn.setMinimumHeight(36)
        self.export_reset_btn.setVisible(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.run_another_btn)
        btn_row.addWidget(self.export_reset_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # ── Public API ────────────────────────────────────────────

    def get_options(self) -> dict[str, bool]:
        return {
            "categories": self.cb_categories.isChecked(),
            "manufacturers": self.cb_manufacturers.isChecked(),
            "products": self.cb_products.isChecked(),
            "customers": self.cb_customers.isChecked(),
            "orders": self.cb_orders.isChecked(),
        }

    def is_wipe(self) -> bool:
        return self.cb_wipe.isChecked()

    def append_log(self, msg: str) -> None:
        self.log_area.append(msg)
        self.log_area.ensureCursorVisible()

    def set_progress(self, label: str, current: int, total: int) -> None:
        self.progress_label.setText(label)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)  # indeterminate

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        for cb in (self.cb_categories, self.cb_manufacturers,
                   self.cb_products, self.cb_customers, self.cb_orders,
                   self.cb_wipe):
            cb.setEnabled(not running)
        # Hide "Run Another" while a migration is in progress — it controls navigation
        if running:
            self.run_another_btn.setVisible(False)
            self.export_reset_btn.setVisible(False)

    def show_run_another(self) -> None:
        self.run_another_btn.setVisible(True)

    def show_export_reset(self) -> None:
        self.export_reset_btn.setVisible(True)

    def reset_for_new_run(self) -> None:
        self.log_area.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self.run_another_btn.setVisible(False)
        self.export_reset_btn.setVisible(False)
