"""Background QThread workers that emit signals back to the GUI."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from migrator.pipeline.importer import DumpImporter, ImportError


class ImportWorker(QThread):
    progress = Signal(int, int)   # bytes_sent, total_bytes
    log = Signal(str)
    finished = Signal(bool, str)  # success, error_msg

    def __init__(
        self,
        importer: DumpImporter,
        sql_file: Path,
        db_name: str,
        parent=None,
    ):
        super().__init__(parent)
        self.importer = importer
        self.sql_file = sql_file
        self.db_name = db_name

    def run(self) -> None:
        try:
            self.log.emit(f"Dropping existing database '{self.db_name}' (if any)…")
            self.importer.drop_database(self.db_name)
            self.log.emit(f"Creating database '{self.db_name}'…")
            self.importer.create_database(self.db_name)
            self.log.emit(f"Loading {self.sql_file.name} — this may take several minutes…")
            self.importer.load_dump(self.sql_file, self.db_name, self._on_progress)
            self.finished.emit(True, "")
        except ImportError as e:
            self.finished.emit(False, str(e))
        except Exception as e:
            self.finished.emit(False, f"Unexpected error: {e}")

    def _on_progress(self, sent: int, total: int) -> None:
        self.progress.emit(sent, total)


class MigrationWorker(QThread):
    log = Signal(str)
    progress = Signal(str, int, int)   # label, current, total
    finished = Signal(bool, str, dict) # success, error_msg, stats

    def __init__(self, runner, parent=None):
        super().__init__(parent)
        self.runner = runner

    def run(self) -> None:
        try:
            # Route runner callbacks through this thread's signals.
            # emit() is thread-safe and queues slots to the receiver's thread.
            self.runner.log = self.log.emit
            self.runner.progress = self.progress.emit
            stats = self.runner.run()
            self.finished.emit(True, "", stats)
        except Exception as e:
            self.finished.emit(False, str(e), {})
