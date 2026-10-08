"""Tests for relaying slash commands directly to the engine (#relay)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.telegram_fakes import FakeTransport
from untether.config import ProjectsConfig
from untether.directives import parse_directives, unescape_relayed_command
from untether.runners.mock import ScriptRunner
from untether.telegram.commands.parse import _parse_slash_command, is_cancel_command
from untether.telegram.commands.relay import (
    BACKEND,
    split_relay_command,
)
from untether.telegram.listen_mode import should_trigger_run
from untether.telegram.types import TelegramIncomingMessage

pytestmark = pytest.mark.anyio


def test_unescape_relayed_command() -> None:
    assert unescape_relayed_command("//compact") == "/compact"
    assert unescape_relayed_command("//model gpt-4o") == "/model gpt-4o"
    assert unescape_relayed_command("///compact") == "//compact"
    assert unescape_relayed_command("//cancel") == "/cancel"
    assert unescape_relayed_command("//") == "/"
    assert unescape_relayed_command("  //compact") == "  /compact"
    assert unescape_relayed_command("/relay /compact") == "/compact"
    assert unescape_relayed_command("/relay compact") == "/compact"
    assert unescape_relayed_command("/relay /model gpt-4o") == "/model gpt-4o"
    assert unescape_relayed_command("normal prompt") == "normal prompt"
    assert unescape_relayed_command("/normal_command") == "/normal_command"
    assert (
        unescape_relayed_command("//compact\nsecond line\nthird line")
        == "/compact\nsecond line\nthird line"
    )


def test_parse_directives_with_relay() -> None:
    projects = ProjectsConfig(projects={}, default_project=None)
    engine_ids = ("claude", "codex", "antigravity")

    # Plain double-slash escape
    res = parse_directives("//compact", engine_ids=engine_ids, projects=projects)
    assert res.prompt == "/compact"
    assert res.engine is None

    # Engine directive + double-slash escape
    res = parse_directives(
        "/claude //compact", engine_ids=engine_ids, projects=projects
    )
    assert res.prompt == "/compact"
    assert res.engine == "claude"

    # Engine directive + /relay
    res = parse_directives(
        "/antigravity /relay /goal", engine_ids=engine_ids, projects=projects
    )
    assert res.prompt == "/goal"
    assert res.engine == "antigravity"

    # Engine directive + /relay without slash
    res = parse_directives(
        "/codex /relay review", engine_ids=engine_ids, projects=projects
    )
    assert res.prompt == "/review"
    assert res.engine == "codex"


def test_parse_slash_command_skips_double_slash() -> None:
    cmd, args = _parse_slash_command("//compact")
    assert cmd is None
    assert args == "//compact"

    cmd, args = _parse_slash_command("//model sonnet")
    assert cmd is None
    assert args == "//model sonnet"

    cmd, args = _parse_slash_command("/relay /compact")
    assert cmd == "relay"
    assert args == "/compact"

    cmd, args = _parse_slash_command("/relay")
    assert cmd == "relay"
    assert args == ""


def test_is_cancel_command_skips_escaped_cancel() -> None:
    assert is_cancel_command("/cancel") is True
    assert is_cancel_command("/cancel@bot") is True
    assert is_cancel_command("//cancel") is False
    assert is_cancel_command("/relay /cancel") is False


def test_should_trigger_run_with_relayed_commands() -> None:
    runtime = MagicMock()
    runtime.available_engine_ids.return_value = ["claude"]
    runtime.project_aliases.return_value = []

    def make_msg(text: str) -> TelegramIncomingMessage:
        return TelegramIncomingMessage(
            transport="telegram",
            chat_id=123,
            message_id=456,
            text=text,
            reply_to_message_id=None,
            reply_to_text=None,
            sender_id=123,
        )

    # In mentions mode, normal unknown commands are ignored
    assert not should_trigger_run(
        make_msg("/unknown_command"),
        bot_username=None,
        runtime=runtime,
        command_ids={"relay"},
        reserved_chat_commands=set(),
    )

    # Double-slash escape triggers the run even in mentions mode
    assert should_trigger_run(
        make_msg("//compact"),
        bot_username=None,
        runtime=runtime,
        command_ids={"relay"},
        reserved_chat_commands=set(),
    )

    # /relay command triggers the run because it is in command_ids
    assert should_trigger_run(
        make_msg("/relay /compact"),
        bot_username=None,
        runtime=runtime,
        command_ids={"relay"},
        reserved_chat_commands=set(),
    )


def test_split_relay_command() -> None:
    assert split_relay_command("relay", "/compact") == "/compact"
    assert split_relay_command("relay", "compact") == "/compact"
    assert split_relay_command("relay", "  /model gpt-4o  ") == "/model gpt-4o"
    assert split_relay_command("relay", "") is None
    assert split_relay_command("relay", "   ") is None
    assert split_relay_command("relay", None) is None
    assert split_relay_command("model", "/compact") is None
    assert split_relay_command(None, "/compact") is None


def test_relay_backend_metadata() -> None:
    assert BACKEND.id == "relay"
    assert BACKEND.description == "Relay a slash command directly to the engine"


async def test_relay_backend_handle() -> None:
    ctx = MagicMock()
    res = await BACKEND.handle(ctx)
    assert res is not None
    assert "/relay <command>" in res.text
    assert "//<command>" in res.text


async def _run_loop_with_msg(
    tmp_path: Path, text: str
) -> tuple[ScriptRunner, FakeTransport]:
    import untether.telegram.loop as telegram_loop
    from tests.telegram_fakes import (
        FakeBot,
        _empty_projects,
        _make_router,
    )
    from untether.markdown import MarkdownPresenter
    from untether.runner_bridge import ExecBridgeConfig
    from untether.runners.mock import Return
    from untether.telegram.bridge import TelegramBridgeConfig
    from untether.transport_runtime import TransportRuntime

    state_path = tmp_path / "untether.toml"
    transport = FakeTransport()
    runner = ScriptRunner([Return(answer="ok")], engine="codex")
    runtime = TransportRuntime(
        router=_make_router(runner),
        projects=_empty_projects(),
        config_path=state_path,
    )
    cfg = TelegramBridgeConfig(
        bot=FakeBot(),
        runtime=runtime,
        chat_id=123,
        startup_msg="",
        exec_cfg=ExecBridgeConfig(
            transport=transport, presenter=MarkdownPresenter(), final_notify=True
        ),
        forward_coalesce_s=0.0,
        media_group_debounce_s=0.0,
        session_mode="chat",
    )
    msg = TelegramIncomingMessage(
        transport="telegram",
        chat_id=123,
        message_id=5,
        text=text,
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=123,
        chat_type="private",
    )

    async def poller(_cfg: TelegramBridgeConfig):
        yield msg

    await telegram_loop.run_main_loop(cfg, poller)
    return runner, transport


async def test_loop_relays_double_slash_command(tmp_path: Path) -> None:
    runner, _ = await _run_loop_with_msg(tmp_path, "//compact")
    assert len(runner.calls) == 1
    prompt, _ = runner.calls[0]
    assert prompt.endswith("/compact")


async def test_loop_relays_slash_relay_command(tmp_path: Path) -> None:
    runner, _ = await _run_loop_with_msg(tmp_path, "/relay /compact")
    assert len(runner.calls) == 1
    prompt, _ = runner.calls[0]
    assert prompt.endswith("/compact")


async def test_loop_bare_relay_shows_usage(tmp_path: Path) -> None:
    runner, transport = await _run_loop_with_msg(tmp_path, "/relay")
    assert len(runner.calls) == 0
    assert len(transport.send_calls) >= 1
    assert "/relay <command>" in transport.send_calls[0]["message"].text
