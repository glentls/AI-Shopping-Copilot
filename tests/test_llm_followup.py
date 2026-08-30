from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.output.followup import FollowUpContext, build_all_missing_ask_message
from src.output.llm_followup import _get_client, build_llm_ask_message


def _ctx(**overrides) -> FollowUpContext:
    defaults = dict(
        scenario="buying",
        n_constraints_known=1,
        exhausted=False,
        turn=2,
        override_seen=False,
        missing_attrs=("color", "size"),
    )
    defaults.update(overrides)
    return FollowUpContext(**defaults)


class GetClientTest(unittest.TestCase):
    def test_raises_runtime_error_when_env_vars_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                _get_client()
            self.assertIn("DOCKER_MODEL_BASE_URL", str(cm.exception))
            self.assertIn("DOCKER_MODEL_API_KEY", str(cm.exception))
            self.assertIn("DOCKER_MODEL_NAME", str(cm.exception))

    def test_raises_when_only_some_env_vars_set(self) -> None:
        env = {"DOCKER_MODEL_BASE_URL": "http://localhost:12434/engines/v1"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                _get_client()
            self.assertIn("DOCKER_MODEL_API_KEY", str(cm.exception))
            self.assertIn("DOCKER_MODEL_NAME", str(cm.exception))
            self.assertNotIn("DOCKER_MODEL_BASE_URL,", str(cm.exception))


class BuildLlmAskMessageFallbackTest(unittest.TestCase):
    """Live LLM behaviour needs Docker Model Runner running and can't be
    exercised in CI/sandbox -- these tests only lock in the never-raise /
    fall-back-to-hardcoded contract, which is what protects the rest of the
    pipeline regardless of whether the LLM call itself succeeds."""

    def test_falls_back_to_hardcoded_message_when_env_vars_missing(self) -> None:
        context = _ctx()
        with patch.dict(os.environ, {}, clear=True):
            result = build_llm_ask_message(context, [])
        self.assertEqual(result, build_all_missing_ask_message(context))

    def test_falls_back_when_openai_client_raises(self) -> None:
        context = _ctx()
        env = {
            "DOCKER_MODEL_BASE_URL": "http://localhost:12434/engines/v1",
            "DOCKER_MODEL_API_KEY": "unused",
            "DOCKER_MODEL_NAME": "does-not-matter",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "src.output.llm_followup._get_client",
                side_effect=ConnectionError("no Docker Model Runner reachable"),
            ):
                result = build_llm_ask_message(context, [])
        self.assertEqual(result, build_all_missing_ask_message(context))

    def test_falls_back_on_empty_llm_response(self) -> None:
        context = _ctx()

        class _FakeMessage:
            content = "   "

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **_kwargs):
                return _FakeResponse()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        with patch(
            "src.output.llm_followup._get_client",
            return_value=(_FakeClient(), "fake-model"),
        ):
            result = build_llm_ask_message(context, [])
        self.assertEqual(result, build_all_missing_ask_message(context))

    def test_never_raises_regardless_of_products_shape(self) -> None:
        # Malformed product dicts (missing "title") must not blow up the
        # prompt-building step -- it should still fall back cleanly.
        context = _ctx()
        with patch.dict(os.environ, {}, clear=True):
            result = build_llm_ask_message(context, [{"parent_asin": "B001"}, {}])
        self.assertEqual(result, build_all_missing_ask_message(context))

    def test_returns_llm_text_on_success(self) -> None:
        context = _ctx()

        class _FakeMessage:
            content = "  I see a few running shoes and hiking boots -- which are you after?  "

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **_kwargs):
                return _FakeResponse()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeClient:
            chat = _FakeChat()

        with patch(
            "src.output.llm_followup._get_client",
            return_value=(_FakeClient(), "fake-model"),
        ):
            result = build_llm_ask_message(
                context, [{"parent_asin": "B001", "title": "Trail Running Shoe"}]
            )
        self.assertEqual(
            result, "I see a few running shoes and hiking boots -- which are you after?"
        )


if __name__ == "__main__":
    unittest.main()
