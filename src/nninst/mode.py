from enum import Enum, auto

__all__ = [
    "ExecutionMode",
    "mode",
    "debug",
    "local",
    "distributed",
    "is_debug",
    "is_local",
    "is_distributed",
    "check",
    "is_check",
]


class ExecutionMode(Enum):
    DEBUG = auto()
    LOCAL = auto()
    DISTRIBUTED = auto()


_mode = ExecutionMode.DISTRIBUTED


def mode():
    return _mode


def debug():
    global _mode
    _mode = ExecutionMode.DEBUG


def local():
    global _mode
    _mode = ExecutionMode.LOCAL


def distributed():
    global _mode
    _mode = ExecutionMode.DISTRIBUTED


def is_debug() -> bool:
    return _mode == ExecutionMode.DEBUG


def is_local() -> bool:
    return _mode == ExecutionMode.LOCAL


def is_distributed() -> bool:
    return _mode == ExecutionMode.DISTRIBUTED


_check = False


def check(flag: bool = True):
    global _check
    _check = flag


def is_check() -> bool:
    return _check
