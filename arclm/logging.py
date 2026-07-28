"""Non-blocking training event logging."""

import json
import logging as _logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


LOG_LEVELS = {
    "quiet": _logging.ERROR,
    "normal": _logging.INFO,
    "verbose": _logging.DEBUG,
    "debug": _logging.DEBUG,
}


def get_logger(name: str = "arclm") -> _logging.Logger:
    """Return an ArcLM package logger without configuring global logging."""

    return _logging.getLogger(name if name.startswith("arclm") else f"arclm.{name}")


def configure_logging(level: str = "normal") -> _logging.Logger:
    """Configure ArcLM package logging for CLI/application entry points."""

    normalized = str(level).lower().strip()
    if normalized not in LOG_LEVELS:
        raise ValueError("level must be one of: " + ", ".join(sorted(LOG_LEVELS)))
    logger = _logging.getLogger("arclm")
    logger.setLevel(LOG_LEVELS[normalized])
    if not logger.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(_logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class AsyncTrainingLogger:
    """Queue-backed logger for console and JSONL training events."""

    def __init__(self, jsonl_path: Optional[str] = None, console: bool = False):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.console = console
        self._queue = queue.Queue()
        self._stop = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def start(self):
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.jsonl_path.open("a", encoding="utf-8")
        if not self._thread.is_alive():
            self._thread.start()
        return self

    def log(self, level: str, event: str, **payload: Any) -> None:
        record = {
            "time": time.time(),
            "level": level.upper(),
            "event": event,
            **payload,
        }
        self._queue.put(record)

    def info(self, event: str, **payload: Any) -> None:
        self.log("INFO", event, **payload)

    def warning(self, event: str, **payload: Any) -> None:
        self.log("WARNING", event, **payload)

    def error(self, event: str, **payload: Any) -> None:
        self.log("ERROR", event, **payload)

    def metric(self, event: str, **payload: Any) -> None:
        self.log("METRIC", event, **payload)

    def close(self):
        self._queue.put(self._stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._file is not None:
            self._file.close()
            self._file = None

    def _run(self):
        while True:
            record = self._queue.get()
            if record is self._stop:
                self._queue.task_done()
                break
            self._write(record)
            self._queue.task_done()

    def _write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        if self.console:
            print(line, file=sys.stderr, flush=True)
