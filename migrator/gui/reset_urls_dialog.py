"""Dialog: collect store URL + output path for the password-reset CSV export."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QDialogButtonBox,
)


class ResetUrlsDialog(QDialog):
    """Modal dialog that returns (store_url, csv_path) on Accept."""

    def __init__(self, parent=None, default_path: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Password-Reset URLs")
        self.setMinimumWidth(520)
        self._build_ui(default_path)

    def _build_ui(self, default_path: Path | None) -> None:
        root = QVBoxLayout(self)

        root.addWidget(QLabel(
            "Generate a one-click password-reset URL for every migrated customer "
            "and write them to a CSV file. Use the CSV with your bulk-mail tool "
            "to invite customers to set a new password."
        ))

        form = QFormLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://shop.example.gr  or  http://localhost/opencart")
        self.url_edit.setToolTip(
            "Storefront URL. Either a clean base (https://shop.example.gr) or the\n"
            "full homepage URL (http://localhost/index.php?route=common/home) works —\n"
            "everything from /index.php onwards is stripped automatically.\n\n"
            "The CSV uses links of the form:\n"
            "  <base>/index.php?route=account/reset&code=<token>"
        )
        form.addRow("Store URL:", self.url_edit)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        default = default_path or (Path.home() / "customer_reset_urls.csv")
        self.path_edit.setText(str(default))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)
        form.addRow("Save CSV to:", path_row)
        root.addLayout(form)

        root.addWidget(QLabel(
            "<i>Customers who already have a reset code in <code>oc_customer.code</code> "
            "(e.g. from a prior \"Forgot Password\" click) will keep their existing code, "
            "so any reset link sitting in their inbox stays valid.</i>"
        ))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save reset-URL CSV", self.path_edit.text(),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _validate_and_accept(self) -> None:
        url = self.url_edit.text().strip()
        path = self.path_edit.text().strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            self.url_edit.setStyleSheet("border: 1px solid #c0392b;")
            self.url_edit.setFocus()
            return
        if not path:
            self.path_edit.setStyleSheet("border: 1px solid #c0392b;")
            self.path_edit.setFocus()
            return
        self.accept()

    def get_values(self) -> tuple[str, Path]:
        return self.url_edit.text().strip(), Path(self.path_edit.text().strip())
