"""Whole-algorithm dispatch; native execution families remain independent."""

from .registry import backend_inventory

__all__ = ["backend_inventory"]
