"""Memory leaf component. Never imports retrieval/, ranking/, or dialog/."""

from .null_memory import distill

__all__ = ["distill"]
