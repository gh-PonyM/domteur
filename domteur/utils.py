import importlib
import inspect
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def module_path(module):
    if isinstance(module, str):
        module = importlib.import_module(module)

    assert module is not None
    return Path(inspect.getfile(module)).parents[0]
