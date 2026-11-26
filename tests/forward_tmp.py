"""Per-test scratch directory inside the checkout's forward namespace.

WP4 outputs are confined to ``.artifacts/forward`` (resolved, no symlinks), so
tests need a real directory there rather than pytest's default basetemp.
"""
import shutil
import uuid

import pytest

from tools import forward_wp4


@pytest.fixture
def tmp_path():
    root = forward_wp4.ROOT / 'pytest-tmp' / uuid.uuid4().hex
    forward_wp4.private_path(root)
    root.mkdir(parents=True)
    forward_wp4.private_path(root)
    try:
        yield root
    finally:
        forward_wp4.private_path(root)
        shutil.rmtree(root)
