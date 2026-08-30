"""Memory leaf component. Never imports retrieval/, ranking/, or dialog/."""

from .distiller import distill

__all__ = ["distill"]
