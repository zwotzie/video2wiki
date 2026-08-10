import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-28s %(message)s"


class _LazyStderrHandler(logging.StreamHandler):
    """StreamHandler that reads sys.stderr on each emit,
    so it works correctly with CliRunner's stderr redirect."""

    def __init__(self) -> None:
        super().__init__(stream=None)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("video2wiki")
    if root.handlers:
        return
    handler = _LazyStderrHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"video2wiki.{name}")
