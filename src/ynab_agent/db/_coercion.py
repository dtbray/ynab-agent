"""Portable coercion helpers for SQL driver result values."""

from decimal import Decimal


def as_int(value: object) -> int:
    """Convert integer-like SQL values, including Decimal scientific notation."""
    return int(Decimal(str(value)))
