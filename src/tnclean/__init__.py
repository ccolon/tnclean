"""tnclean - Transport Network Cleaner

A small Python package to clean & simplify transport network lines.
"""

from .pipeline import clean_network
from .types import CleanConfig, DedupPolicy, PrecisionMode, SimplifyMode

__version__ = "0.1.0"
__all__ = [
    "clean_network",
    "CleanConfig", 
    "DedupPolicy",
    "PrecisionMode",
    "SimplifyMode",
]