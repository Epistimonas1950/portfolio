"""Load a phpMyAdmin SQL dump into a local MySQL/MariaDB database."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Callable, Optional


class ImportError(Exception):
    pass


class DumpImporter:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def _base_cmd(self) -> list[str]:
        cmd = [
            "mysql", f"-h{self.host}", f"-P{self.port}", f"-u{self.user}",
            "--max-allowed-packet=1G", "--default-character-set=utf8mb4",
        ]
        if self.password:
            cmd.append(f"-p{self.password}")
        return cmd

    def test_connection(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                self._base_cmd() + ["-e", "SELECT 1"],
                capture_output=True, timeout=10,
            )
            return r.returncode == 0, r.stderr.decode().strip()
        except FileNotFoundError:
            return False, "'mysql' command not found — install MySQL client tools."
        except subprocess.TimeoutExpired:
            return False, "Connection timed out."

    def create_database(self, db_name: str) -> None:
        sql = (
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        r = subprocess.run(self._base_cmd() + ["-e", sql], capture_output=True)
        if r.returncode != 0:
            raise ImportError(r.stderr.decode().strip())

    def drop_database(self, db_name: str) -> None:
        subprocess.run(
            self._base_cmd() + ["-e", f"DROP DATABASE IF EXISTS `{db_name}`"],
            capture_output=True,
        )

    def load_dump(
        self,
        sql_file: Path,
        db_name: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Stream sql_file bytes into db_name; calls progress_cb(bytes_sent, total)."""
        total_bytes = sql_file.stat().st_size
        cmd = self._base_cmd() + [db_name]

        with open(sql_file, "rb") as fh:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk = 64 * 1024  # 64 KB
            sent = 0
            pipe_broke = False
            try:
                while True:
                    data = fh.read(chunk)
                    if not data:
                        break
                    try:
                        proc.stdin.write(data)
                    except BrokenPipeError:
                        # mysql died mid-stream — stop sending and let stderr explain
                        pipe_broke = True
                        break
                    sent += len(data)
                    if progress_cb:
                        progress_cb(sent, total_bytes)
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            proc.wait()

        if proc.returncode != 0 or pipe_broke:
            err = proc.stderr.read().decode(errors="replace").strip()
            raise ImportError(err or "mysql client exited unexpectedly")
