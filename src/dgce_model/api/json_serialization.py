"""Helpers for converting DGCE outputs into JSON-friendly structures."""
from __future__ import annotations

import json
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

import numpy as np


def json_default(obj: Any) -> Any:
    """Fallback encoder for numpy/pandas objects."""

    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json_ready(data: Any) -> Any:
    """Return a structure composed of JSON-serializable primitives."""

    return json.loads(json.dumps(data, default=json_default))


__all__ = ["json_default", "to_json_ready"]

