"""
Minimal stubs for OpenFisca core Variable API to allow import.
This is a placeholder for openfisca_core.model_api.
"""
class Variable:
    pass

YEAR = None

def Enum(name, vals):
    """Create dummy Enum type with attributes for each value."""
    attrs = {}
    for v in vals:
        key = v.decode() if isinstance(v, bytes) else v
        attrs[key] = v
    return type(name, (), attrs)

def select(conds, vals, default=None):
    return default

def where(cond, a, b):
    return a

def not_(x):
    return not x

def max_(a, b):
    return max(a, b)