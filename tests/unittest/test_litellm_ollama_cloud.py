"""
Tests for using PR-Agent against an authenticated Ollama endpoint (e.g. Ollama Cloud).

Verifies the config-only path that lets a hosted Ollama model work end-to-end:
  - the `ollama_chat/` model and `[ollama] api_base` reach the acompletion call,
  - custom headers (bearer token + cookie) from LITELLM.EXTRA_HEADERS are forwarded,
  - the `think` field from LITELLM.EXTRA_BODY is forwarded,
  - both extra_headers and extra_body accept an already-parsed dict (the env-var / GitHub
    Action path where Dynaconf may deserialize the value) as well as a JSON string.
"""
from unittest.mock import AsyncMock, patch

import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.litellm_helpers import _process_litellm_extra_body


class FakeBox:
    def __init__(self, values=None, **attrs):
        self._values = values or {}
        for key, value in attrs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._values.get(key, default)


class FakeSettings:
    def __init__(self, litellm_attrs=None, settings_values=None):
        self.config = FakeBox(
            {"seed": -1},
            reasoning_effort=None,
            ai_timeout=30,
            custom_reasoning_model=False,
            max_model_tokens=128000,
            verbosity_level=0,
            model="ollama_chat/glm-4.6:cloud",
        )
        self.litellm = FakeBox(**(litellm_attrs or {}))
        self._settings_values = settings_values or {}
        # __init__ reads get_settings().ollama.api_base after the OLLAMA.API_BASE guard passes.
        self.ollama = FakeBox(api_base=self._settings_values.get("OLLAMA.API_BASE"))

    def get(self, key, default=None):
        return self._settings_values.get(key, default)


def _mock_response():
    from unittest.mock import MagicMock
    mock = MagicMock()
    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    mock.__getitem__.side_effect = response.__getitem__
    mock.dict.return_value = response
    return mock


async def _run_and_capture(monkeypatch, litellm_attrs, settings_values):
    settings = FakeSettings(litellm_attrs=litellm_attrs, settings_values=settings_values)
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)
    # _process_litellm_extra_body pulls get_settings() from the helpers module too.
    monkeypatch.setattr("pr_agent.algo.ai_handlers.litellm_helpers.get_settings", lambda: settings)

    with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_response()
        handler = litellm_handler.LiteLLMAIHandler()
        await handler.chat_completion(
            model="ollama_chat/glm-4.6:cloud", system="sys", user="usr", temperature=0.2
        )
    return mock_call.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("as_dict", [False, True])
async def test_ollama_cloud_headers_body_and_api_base(monkeypatch, as_dict):
    """Full path: model, api_base, auth+cookie headers and think all reach acompletion."""
    headers = {"Authorization": "Bearer access-token", "Cookie": "aid=cookie-id"}
    body = {"think": True}
    litellm_attrs = {
        # str form mirrors .secrets.toml; dict form mirrors the env-var / GitHub Action path.
        "extra_headers": headers if as_dict else '{"Authorization": "Bearer access-token", "Cookie": "aid=cookie-id"}',
        "extra_body": body if as_dict else '{"think": true}',
    }
    settings_values = {
        "OLLAMA.API_BASE": "https://ollama.com",
        "LITELLM.EXTRA_HEADERS": litellm_attrs["extra_headers"],
    }

    kwargs = await _run_and_capture(monkeypatch, litellm_attrs, settings_values)

    assert kwargs["model"] == "ollama_chat/glm-4.6:cloud"
    assert kwargs["api_base"] == "https://ollama.com"
    assert kwargs["extra_headers"] == headers
    assert kwargs["extra_headers"]["Authorization"] == "Bearer access-token"
    assert kwargs["extra_headers"]["Cookie"] == "aid=cookie-id"
    assert kwargs["think"] is True


@pytest.mark.parametrize("as_dict", [False, True])
def test_think_is_an_allowed_extra_body_key(monkeypatch, as_dict):
    settings = FakeSettings(litellm_attrs={"extra_body": {"think": True} if as_dict else '{"think": true}'})
    monkeypatch.setattr("pr_agent.algo.ai_handlers.litellm_helpers.get_settings", lambda: settings)

    result = _process_litellm_extra_body({"model": "ollama_chat/glm-4.6:cloud"})

    assert result["think"] is True


def test_unsupported_extra_body_key_is_rejected(monkeypatch):
    settings = FakeSettings(litellm_attrs={"extra_body": '{"not_a_real_field": 1}'})
    monkeypatch.setattr("pr_agent.algo.ai_handlers.litellm_helpers.get_settings", lambda: settings)

    with pytest.raises(ValueError, match="unsupported keys"):
        _process_litellm_extra_body({"model": "ollama_chat/glm-4.6:cloud"})
