from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from untether.model import DiscoveredModel
from untether.runners.claude import ClaudeRunner, fetch_claude_models
from untether.runners.codex import CodexRunner, parse_codex_models
from untether.runners.discovery import (
    fetch_available_models,
    get_cached_models,
    get_model_effort_levels,
    model_supports_effort,
    reset_model_cache,
    set_cached_models,
)
from untether.runners.opencode import OpenCodeRunner, parse_opencode_models
from untether.runners.pi import PiRunner, parse_pi_models


def test_discovered_model_properties() -> None:
    m1 = DiscoveredModel(model_id="test-model", effort_levels=("low", "high"))
    assert m1.supports_effort is True
    assert m1.effort_levels == ("low", "high")

    m2 = DiscoveredModel(model_id="test-model-2", effort_levels=())
    assert m2.supports_effort is False


def test_parse_codex_models() -> None:
    payload = json.dumps(
        {
            "models": [
                {
                    "slug": "gpt-5-codex",
                    "visibility": "list",
                    "supported_reasoning_levels": [
                        {"effort": "low"},
                        {"effort": "medium"},
                        {"effort": "high"},
                    ],
                },
                {
                    "slug": "gpt-4.5-turbo",
                    "visibility": "hidden",
                },
                {
                    "slug": "codex-mini",
                    "visibility": "list",
                    "supported_reasoning_levels": [],
                },
            ]
        }
    )
    models = parse_codex_models(payload)
    assert len(models) == 2
    assert models[0].model_id == "gpt-5-codex"
    assert models[0].effort_levels == ("low", "medium", "high")
    assert models[0].supports_effort is True
    assert models[1].model_id == "codex-mini"
    assert models[1].supports_effort is False

    # Corrupt or invalid JSON falls back gracefully
    assert parse_codex_models("{invalid json") == []
    assert parse_codex_models(b"not json") == []
    assert parse_codex_models(json.dumps({"models": "not a list"})) == []


def test_parse_opencode_models() -> None:
    sample_output = """
anthropic/claude-3-7-sonnet
openai/gpt-4o
openai/o3-mini
deepseek/deepseek-r1
"""
    models = parse_opencode_models(sample_output)
    assert len(models) == 4
    assert models[0].model_id == "anthropic/claude-3-7-sonnet"
    assert "claude-3-7-sonnet" in models[0].raw_ids
    assert models[2].model_id == "openai/o3-mini"
    assert "o3-mini" in models[2].raw_ids

    assert parse_opencode_models("") == []


def test_parse_pi_models() -> None:
    sample_output = """
pi-coding-v1
pi-fast-v2
# a comment line
"""
    models = parse_pi_models(sample_output)
    assert len(models) == 2
    assert models[0].model_id == "pi-coding-v1"
    assert models[1].model_id == "pi-fast-v2"

    assert parse_pi_models("") == []


@pytest.mark.anyio
async def test_fetch_claude_models() -> None:
    models = await fetch_claude_models()
    assert len(models) > 0
    sonnet_5 = next((m for m in models if m.model_id == "claude-sonnet-5-5"), None)
    assert sonnet_5 is not None
    assert sonnet_5.supports_effort is True
    assert "medium" in sonnet_5.effort_levels


@pytest.mark.anyio
async def test_discovery_ttl_caching() -> None:
    reset_model_cache()
    try:
        dummy = [DiscoveredModel(model_id="m1")]
        set_cached_models(dummy, engine="codex", timestamp=100.0)
        assert get_cached_models("codex") == dummy

        # Cache lookup
        with patch("untether.runners.discovery.time.monotonic", return_value=200.0):
            res = await fetch_available_models("codex")
            assert res == dummy

        # Expired lookup triggers fetch
        with (
            patch("untether.runners.discovery.time.monotonic", return_value=1000.0),
            patch(
                "untether.runners.codex.fetch_codex_models",
                new_callable=AsyncMock,
                return_value=[DiscoveredModel(model_id="fresh")],
            ) as mock_fetch,
        ):
            res2 = await fetch_available_models("codex")
            assert res2[0].model_id == "fresh"
            mock_fetch.assert_called_once()

        # Engine-specific reset
        set_cached_models(dummy, engine="claude")
        assert len(get_cached_models("claude")) == 1
        reset_model_cache("claude")
        assert len(get_cached_models("claude")) == 0
    finally:
        reset_model_cache()


@pytest.mark.anyio
async def test_discovery_runner_delegation() -> None:
    reset_model_cache()
    try:
        mock_runner = MagicMock()
        mock_runner.fetch_available_models = AsyncMock(
            return_value=[DiscoveredModel(model_id="custom-runner-model")]
        )
        res = await fetch_available_models("custom_engine", runner=mock_runner)
        assert len(res) == 1
        assert res[0].model_id == "custom-runner-model"
    finally:
        reset_model_cache()


@pytest.mark.anyio
async def test_get_model_effort_levels_and_supports_effort() -> None:
    reset_model_cache()
    try:
        models = [
            DiscoveredModel(
                model_id="gpt-5-codex",
                effort_levels=("low", "medium", "high", "xhigh"),
                raw_ids=("gpt-5-codex",),
            ),
            DiscoveredModel(
                model_id="codex-mini",
                effort_levels=(),
                raw_ids=("codex-mini",),
            ),
        ]
        set_cached_models(models, engine="codex")

        efforts = await get_model_effort_levels("codex", "gpt-5-codex")
        assert efforts == ("low", "medium", "high", "xhigh")
        assert model_supports_effort("codex", "gpt-5-codex") is True

        no_efforts = await get_model_effort_levels("codex", "codex-mini")
        assert no_efforts == ()
        assert model_supports_effort("codex", "codex-mini") is False

        # OpenCode does not support reasoning
        assert model_supports_effort("opencode", "anthropic/claude-3-7-sonnet") is False
    finally:
        reset_model_cache()


@pytest.mark.anyio
async def test_runner_classes_implement_fetch_available_models() -> None:
    claude = ClaudeRunner(claude_cmd="claude")
    claude_models = await claude.fetch_available_models()
    assert len(claude_models) > 0

    codex = CodexRunner(codex_cmd="codex")
    with patch(
        "untether.runners.codex.fetch_codex_models",
        new_callable=AsyncMock,
        return_value=[DiscoveredModel(model_id="mock-codex")],
    ):
        codex_models = await codex.fetch_available_models()
        assert len(codex_models) == 1

    opencode = OpenCodeRunner(opencode_cmd="opencode")
    with patch(
        "untether.runners.opencode.fetch_opencode_models",
        new_callable=AsyncMock,
        return_value=[DiscoveredModel(model_id="mock-opencode")],
    ):
        opencode_models = await opencode.fetch_available_models()
        assert len(opencode_models) == 1

    pi = PiRunner(pi_cmd="pi")
    with patch(
        "untether.runners.pi.fetch_pi_models",
        new_callable=AsyncMock,
        return_value=[DiscoveredModel(model_id="mock-pi")],
    ):
        pi_models = await pi.fetch_available_models()
        assert len(pi_models) == 1


@pytest.mark.anyio
async def test_render_model_view_multi_engine() -> None:
    from tests.telegram_fakes import FakeTransport, make_cfg
    from untether.telegram.commands.model import _render_model_view
    from untether.telegram.types import TelegramIncomingMessage

    reset_model_cache()
    try:
        set_cached_models(
            [
                DiscoveredModel(model_id="gpt-5-codex", effort_levels=("low", "high")),
                DiscoveredModel(model_id="codex-mini", effort_levels=()),
            ],
            engine="codex",
        )
        transport = FakeTransport()
        cfg = make_cfg(transport, engine_id="codex")
        msg = TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=1,
            text="/model",
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=456,
        )

        text, reply_markup = await _render_model_view(
            cfg,
            msg,
            ambient_context=None,
            topic_store=None,
            chat_prefs=None,
            tkey=None,
        )
        assert "available models (codex):" in text
        assert "• gpt-5-codex (effort: low, high)" in text
        assert "• codex-mini" in text
        assert reply_markup is not None
        kb = reply_markup["inline_keyboard"]
        buttons = [btn["text"] for row in kb for btn in row]
        assert "gpt-5-codex" in buttons
        assert "codex-mini" in buttons
        assert "🔄 Refresh" in buttons
    finally:
        reset_model_cache()


@pytest.mark.anyio
async def test_codex_model_callback_shows_effort_selector() -> None:
    from tests.telegram_fakes import FakeTransport, make_cfg
    from untether.telegram.commands.model import (
        _handle_callback_model,
        model_selector_state_machine,
    )
    from untether.telegram.types import TelegramCallbackQuery

    reset_model_cache()
    try:
        set_cached_models(
            [
                DiscoveredModel(
                    model_id="gpt-5-codex", effort_levels=("low", "medium", "high")
                )
            ],
            engine="codex",
        )
        transport = FakeTransport()
        cfg = make_cfg(transport, engine_id="codex")
        query = TelegramCallbackQuery(
            transport="telegram",
            chat_id=123,
            message_id=10,
            callback_query_id="q1",
            data="model:set:gpt-5-codex",
            sender_id=456,
        )

        await _handle_callback_model(
            cfg,
            query,
            "set:gpt-5-codex",
            ambient_context=None,
            topic_store=None,
            chat_prefs=None,
        )

        pending = model_selector_state_machine.get_pending(123, 10)
        assert pending is not None
        assert pending.selected_model == "gpt-5-codex"
        assert pending.effort_levels == ("low", "medium", "high")

        # Transport edited the message to show effort buttons
        assert len(transport.edit_calls) == 1
        edit = transport.edit_calls[0]
        assert "Select effort for gpt-5-codex:" in edit["message"].text
        kb = edit["message"].extra["reply_markup"]["inline_keyboard"]
        labels = [btn["text"] for row in kb for btn in row]
        assert "Low" in labels
        assert "Medium" in labels
        assert "High" in labels
    finally:
        reset_model_cache()


@pytest.mark.anyio
async def test_opencode_model_callback_sets_without_effort(tmp_path) -> None:
    from pathlib import Path

    from tests.telegram_fakes import FakeTransport, make_cfg
    from untether.telegram.chat_prefs import ChatPrefsStore
    from untether.telegram.commands.model import (
        _handle_callback_model,
        model_selector_state_machine,
    )
    from untether.telegram.types import TelegramCallbackQuery

    reset_model_cache()
    try:
        set_cached_models(
            [DiscoveredModel(model_id="anthropic/claude-3-7-sonnet")],
            engine="opencode",
        )
        prefs_path = Path(tmp_path) / "chat_prefs.json"
        chat_prefs = ChatPrefsStore(prefs_path)
        transport = FakeTransport()
        cfg = make_cfg(transport, engine_id="opencode")
        query = TelegramCallbackQuery(
            transport="telegram",
            chat_id=123,
            message_id=10,
            callback_query_id="q2",
            data="model:set:anthropic/claude-3-7-sonnet",
            sender_id=456,
        )

        await _handle_callback_model(
            cfg,
            query,
            "set:anthropic/claude-3-7-sonnet",
            ambient_context=None,
            topic_store=None,
            chat_prefs=chat_prefs,
        )

        # No effort state is queued for OpenCode
        assert model_selector_state_machine.get_pending(123, 10) is None

        # Override saved to chat prefs
        override = await chat_prefs.get_engine_override(123, "opencode")
        assert override is not None
        assert override.model == "anthropic/claude-3-7-sonnet"
    finally:
        reset_model_cache()
