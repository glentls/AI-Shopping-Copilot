from __future__ import annotations

from dataclasses import fields, replace

import pytest

from src.contracts.config import CONFIGS, SUBMISSION_CONFIG_NAME, get_run_config


def test_config_k_is_t_with_only_no_repeat_enabled() -> None:
    t = CONFIGS["T"]
    k = CONFIGS["K"]
    differing = {
        field.name
        for field in fields(k)
        if getattr(k, field.name) != getattr(t, field.name)
    }

    assert differing == {"name", "exclude_shown"}
    assert k == replace(t, name="K", exclude_shown=True)


def test_config_k_preserves_t_provenance_sensitive_controls() -> None:
    k = CONFIGS["K"]

    assert k.dense_text_recipe == "full"
    assert k.negative_preference is False
    assert k.rerank_window == 0
    assert k.rerank_window_scope == "all"


def test_config_k_is_opt_in_and_does_not_change_submission_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SHOPLENS_CONFIG", raising=False)

    assert SUBMISSION_CONFIG_NAME == "T"
    assert get_run_config() is CONFIGS["T"]
    assert get_run_config("K") is CONFIGS["K"]
