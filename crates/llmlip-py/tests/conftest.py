"""Optional external reference models are never bundled with the package."""
from pathlib import Path
import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        fn = getattr(item, "obj", None)
        if fn is None or not hasattr(fn, "__code__"):
            continue
        for name in fn.__code__.co_names:
            value = fn.__globals__.get(name)
            if name.endswith("MODEL_PATH") and isinstance(value, Path) and not value.exists():
                item.add_marker(pytest.mark.skip(reason="optional external reference model unavailable: " + name))
