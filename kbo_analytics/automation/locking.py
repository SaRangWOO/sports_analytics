from __future__ import annotations

import fcntl
import time
from pathlib import Path


class LockUnavailable(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: Path, timeout_seconds: int = 0):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    self.handle = None
                    raise LockUnavailable(f"lock unavailable: {self.path}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, traceback):
        if self.handle:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
