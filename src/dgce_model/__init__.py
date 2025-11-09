"""UFISCAL – OG-Core + OpenFisca integration package for fiscal analytics."""

from importlib import metadata

try:  # pragma: no cover - fallback when package not installed
    __version__ = metadata.version("ufiscal")
except metadata.PackageNotFoundError:  # type: ignore[attr-defined]
    __version__ = "0.0.0"

__all__ = ["__version__"]
