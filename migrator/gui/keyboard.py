"""Make Enter/Return in single-line inputs advance focus to the next field."""
from __future__ import annotations

from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication, QLineEdit, QSpinBox


class EnterAdvancesFocus(QObject):
    """Application-wide event filter: Enter in a QLineEdit/QSpinBox shifts focus
    to the next widget in tab order. Other widgets keep their normal Enter behavior
    (so QPushButton.click still fires when a button has focus)."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Return, Qt.Key.Key_Enter,
        ):
            widget = QApplication.focusWidget()
            if isinstance(widget, (QLineEdit, QSpinBox)):
                widget.focusNextChild()
                return True
        return False
