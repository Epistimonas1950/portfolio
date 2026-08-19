"""Modal dialog showing preflight check results + auto-fix button."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)

from migrator.pipeline.preflight import PreflightResult


class PreflightDialog(QDialog):
    """Modal dialog. After exec(), `choice` is one of: 'cancel' | 'fix' | 'continue'."""

    def __init__(self, result: PreflightResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preflight Check")
        self.setMinimumWidth(540)
        self.result = result
        self.choice = "cancel"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(QLabel("<h3>Preflight Check Results</h3>"))

        if not self.result.connection_ok:
            layout.addWidget(QLabel(
                "<span style='color:#e74c3c'>✗ Cannot connect to target database</span>"
            ))
            layout.addWidget(QLabel(self.result.error))
        else:
            layout.addWidget(self._line(True, "Database connection OK"))

            if self.result.missing_tables:
                layout.addWidget(self._line(
                    False,
                    f"Missing {len(self.result.missing_tables)} tables: "
                    f"{', '.join(self.result.missing_tables[:3])}…",
                    fatal=True,
                ))
            else:
                layout.addWidget(self._line(True, "All required tables present"))

            if self.result.myisam_tables:
                layout.addWidget(self._line(
                    False,
                    f"{len(self.result.myisam_tables)} tables use MyISAM "
                    "(no transaction rollback — migrations cannot recover from errors)",
                ))
            else:
                layout.addWidget(self._line(True, "All target tables on InnoDB"))

            if self.result.utf8mb3_tables:
                layout.addWidget(self._line(
                    False,
                    f"{len(self.result.utf8mb3_tables)} tables use utf8mb3 "
                    "(blocks 4-byte UTF-8 — emojis fail with DataError 1366)",
                ))
            else:
                layout.addWidget(self._line(True, "All target tables on utf8mb4"))

            if self.result.language_count == 0:
                layout.addWidget(self._line(
                    False, "No languages defined in OpenCart", fatal=True,
                ))
            else:
                layout.addWidget(self._line(
                    True, f"{self.result.language_count} language(s) defined",
                ))

        layout.addStretch()

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()

        if self.result.is_fixable:
            fix_btn = QPushButton("🛠  Auto-Fix Issues")
            fix_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px 14px; }")
            fix_btn.clicked.connect(self._fix)
            btn_row.addWidget(fix_btn)

        if not self.result.is_fatal:
            continue_btn = QPushButton("Continue →")
            continue_btn.clicked.connect(self._continue)
            btn_row.addWidget(continue_btn)

        layout.addLayout(btn_row)

    def _line(self, ok: bool, msg: str, fatal: bool = False) -> QLabel:
        icon = "✓" if ok else ("✗" if fatal else "⚠")
        color = "#27ae60" if ok else ("#e74c3c" if fatal else "#e67e22")
        return QLabel(
            f"<span style='color:{color}; font-size:14px;'>{icon}</span>  {msg}"
        )

    def _cancel(self) -> None:
        self.choice = "cancel"
        self.reject()

    def _fix(self) -> None:
        self.choice = "fix"
        self.accept()

    def _continue(self) -> None:
        self.choice = "continue"
        self.accept()
