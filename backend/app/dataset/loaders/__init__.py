"""Loaders package."""

from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.loaders.fakeddit import FakedditLoader
from backend.app.dataset.loaders.fakenewsnet import FakeNewsNetLoader
from backend.app.dataset.loaders.liar import LiarDatasetLoader
from backend.app.dataset.loaders.nela import NelaGtLoader

__all__ = [
    "BaseDatasetLoader",
    "LiarDatasetLoader",
    "FakeNewsNetLoader",
    "FakedditLoader",
    "NelaGtLoader",
]
