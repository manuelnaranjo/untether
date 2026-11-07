"""Unit and integration tests for the /plan command."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import untether.telegram.loop as telegram_loop
from tests.telegram_fakes import (
    FakeBot,
    FakeTransport,
    _empty_projects,
    _make_router,
)
from tests.test_command_engine_gates import FakeCommandContext, FakeTransportRuntime
from untether.commands import get_command
from untether.context import RunContext
from untether.markdown import MarkdownPresenter
from untether.model import UntetherEvent
from untether.runner_bridge import ExecBridgeConfig
from untether.runners.mock import Return, ScriptRunner
from untether.runners.run_options import get_run_options
from untether.telegram.bridge import TelegramBridgeConfig
from untether.telegram.chat_prefs import ChatPrefsStore, resolve_prefs_path
from untether.telegram.commands.menu import build_bot_commands
from untether.telegram.commands.overrides import SETTINGS_ADMIN_GATE
from untether.telegram.commands.plan import (
    ActionPlan,
    PlanCommand,
    split_plan_command,
)
from untether.telegram.loop import _merge_block_reason
from untether.telegram.types import TelegramIncomingMessage
from untether.transport_runtime import TransportRuntime


class RecordingRunner(ScriptRunner):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.recorded_options: list[Any] = []

    async def run(self, prompt: str, resume: Any) -> AsyncIterator[UntetherEvent]:
        self.recorded_options.append(get_run_options())
        async for evt in super().run(prompt, resume):
            yield evt


def test_split_plan_command() -> None:
    # Non-plan command ids
    assert split_plan_command(None, "task") is None
    assert split_plan_command("planmode", "task") is None
    assert split_plan_command("model", "task") is None

    # Bare / whitespace
    assert split_plan_command("plan", None) is None
    assert split_plan_command("plan", "") is None
    assert split_plan_command("plan", "   ") is None

    # Setting subcommands (case-insensitive)
    assert split_plan_command("plan", "on") is None
    assert split_plan_command("plan", "ON") is None
    assert split_plan_command("plan", "off") is None
    assert split_plan_command("plan", "OFF") is None
    assert split_plan_command("plan", "show") is None
    assert split_plan_command("plan", "SHOW") is None
    assert split_plan_command("plan", "clear") is None
    assert split_plan_command("plan", "CLEAR") is None
    assert split_plan_command("plan", "auto") is None
    assert split_plan_command("plan", "plan-auto") is None
    assert split_plan_command("plan", "plan") is None

    # Prompt tasks (single-word, multi-word, multi-line, starting with setting words)
    assert split_plan_command("plan", "refactor") == ("plan", "refactor")
    assert split_plan_command("plan", "build a website") == ("plan", "build a website")
    assert split_plan_command("plan", "show me the database schema") == (
        "plan",
        "show me the database schema",
    )
    assert split_plan_command("plan", "onto the roof") == ("plan", "onto the roof")
    assert split_plan_command("plan", "clear the cache") == ("plan", "clear the cache")
    assert split_plan_command("plan", "write a unit test\nwith pytest") == (
        "plan",
        "write a unit test\nwith pytest",
    )


def test_plan_admin_gate() -> None:
    # Mode-mutating settings invocations require admin
    assert PlanCommand.admin_gate("") == SETTINGS_ADMIN_GATE
    assert PlanCommand.admin_gate("on") == SETTINGS_ADMIN_GATE
    assert PlanCommand.admin_gate("off") == SETTINGS_ADMIN_GATE
    assert PlanCommand.admin_gate("clear") == SETTINGS_ADMIN_GATE
    assert PlanCommand.admin_gate("auto") == SETTINGS_ADMIN_GATE
    assert PlanCommand.admin_gate("plan-auto") == SETTINGS_ADMIN_GATE

    # Prompts and non-mutating subcommands remain open
    assert PlanCommand.admin_gate("show") is None
    assert PlanCommand.admin_gate("implement login feature") is None
    assert PlanCommand.admin_gate("refactor") is None


def test_plan_description_and_menu() -> None:
    cmd = PlanCommand()
    assert len(cmd.description) <= 256
    assert "Plan" in cmd.description

    backend = get_command("plan", allowlist=None, required=False)
    assert isinstance(backend, PlanCommand)

    runtime = TransportRuntime(
        router=_make_router(ScriptRunner([Return(answer="ok")], engine="antigravity")),
        projects=_empty_projects(),
    )
    commands = build_bot_commands(runtime)
    assert {"command": "plan", "description": cmd.description} in commands


@pytest.mark.anyio
async def test_plan_unsupported_engine(tmp_path: Path) -> None:
    ctx = FakeCommandContext(
        command="plan",
        args_text="show",
        config_path=tmp_path / "untether.toml",
        runtime=FakeTransportRuntime(default_engine="codex"),
    )
    res = await PlanCommand().handle(ctx)  # type: ignore[arg-type]
    assert res is not None
    assert "Plan mode is only available for Antigravity and Claude Code." in res.text
    assert "<b>codex</b>" in res.text
    assert "Use /config → Approval policy instead." in res.text


@pytest.mark.anyio
async def test_plan_no_config_path() -> None:
    ctx = FakeCommandContext(
        command="plan",
        args_text="on",
        config_path=None,
    )
    res = await PlanCommand().handle(ctx)  # type: ignore[arg-type]
    assert res is not None
    assert "permission mode overrides unavailable (no config path)." in res.text


@pytest.mark.anyio
async def test_plan_antigravity_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "untether.toml"

    def _prefs() -> ChatPrefsStore:
        return ChatPrefsStore(resolve_prefs_path(config_path))

    async def _run(args: str, toml_mode: str | None = None) -> str:
        runner = SimpleNamespace(default_permission_mode=toml_mode)
        ctx = FakeCommandContext(
            command="plan",
            args_text=args,
            config_path=config_path,
            runtime=FakeTransportRuntime(default_engine="antigravity", runner=runner),
        )
        res = await PlanCommand().handle(ctx)  # type: ignore[arg-type]
        assert res is not None
        return res.text

    # Show default
    text = await _run("show")
    assert "<b>workspace</b>" in text.lower()

    # Turn on
    text = await _run("on")
    assert "<b>plan first</b>" in text.lower()
    override = await _prefs().get_engine_override(100, "antigravity")
    assert override is not None and override.permission_mode == "plan"

    # Toggle off
    text = await _run("")
    assert "<b>workspace</b>" in text.lower()
    override = await _prefs().get_engine_override(100, "antigravity")
    assert override is None or override.permission_mode is None

    # Toggle on
    text = await _run("")
    assert "<b>plan first</b>" in text.lower()

    # Turn off explicitly
    text = await _run("off")
    assert "<b>workspace</b>" in text.lower()

    # Clear override
    text = await _run("clear")
    assert "override cleared" in text

    # Auto / plan-auto are refused on Antigravity
    text = await _run("auto")
    assert "isn't available for Antigravity" in text
    text = await _run("plan-auto")
    assert "isn't available for Antigravity" in text

    # Off when TOML default is plan sets workspace override
    text = await _run("off", toml_mode="plan")
    override = await _prefs().get_engine_override(100, "antigravity")
    assert override is not None and override.permission_mode == "workspace"


@pytest.mark.anyio
async def test_plan_claude_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "untether.toml"

    def _prefs() -> ChatPrefsStore:
        return ChatPrefsStore(resolve_prefs_path(config_path))

    async def _run(args: str) -> str:
        ctx = FakeCommandContext(
            command="plan",
            args_text=args,
            config_path=config_path,
            runtime=FakeTransportRuntime(default_engine="claude"),
        )
        res = await PlanCommand().handle(ctx)  # type: ignore[arg-type]
        assert res is not None
        return res.text

    # Show default
    text = await _run("show")
    assert "no override" in text

    # Turn on
    text = await _run("on")
    assert "<b>on</b>" in text
    assert "--permission-mode plan" in text
    override = await _prefs().get_engine_override(100, "claude")
    assert override is not None and override.permission_mode == "plan"

    # Toggle off
    text = await _run("")
    assert "<b>off</b>" in text
    override = await _prefs().get_engine_override(100, "claude")
    assert override is not None and override.permission_mode == "acceptEdits"

    # Set plan-auto
    text = await _run("plan-auto")
    assert "<b>plan-auto</b>" in text

    # Clear
    text = await _run("clear")
    assert "override cleared" in text
    override = await _prefs().get_engine_override(100, "claude")
    assert override is None or override.permission_mode is None


@pytest.mark.anyio
async def test_plan_prompt_with_executor(tmp_path: Path) -> None:
    config_path = tmp_path / "untether.toml"
    executed_requests: list[Any] = []

    class FakeExecutor:
        async def run_one(self, req: Any) -> None:
            executed_requests.append(req)

    ctx = FakeCommandContext(
        command="plan",
        args_text="refactor the payment flow",
        config_path=config_path,
        runtime=FakeTransportRuntime(default_engine="antigravity"),
        executor=FakeExecutor(),
    )
    res = await PlanCommand().handle(ctx)  # type: ignore[arg-type]
    assert res is None  # Handled by executor
    assert len(executed_requests) == 1
    req = executed_requests[0]
    assert req.prompt == "refactor the payment flow"
    assert req.engine == "antigravity"
    assert req.context == RunContext(permission_mode="plan")


def _msg(text: str, message_id: int = 5, **kw: Any) -> TelegramIncomingMessage:
    return TelegramIncomingMessage(
        transport="telegram",
        chat_id=123,
        message_id=message_id,
        text=text,
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=123,
        chat_type="private",
        **kw,
    )


@pytest.mark.anyio
async def test_loop_plan_prompt_antigravity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "untether.toml"
    transport = FakeTransport()
    runner = RecordingRunner([Return(answer="Plan created")], engine="antigravity")
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

    steer_calls: list[dict[str, Any]] = []

    async def fake_maybe_steer(cfg: Any, **kw: Any) -> bool:
        steer_calls.append(kw)
        return False

    monkeypatch.setattr(telegram_loop, "maybe_steer", fake_maybe_steer)

    async def poller(_cfg: TelegramBridgeConfig):
        yield _msg("/plan design the database")

    await telegram_loop.run_main_loop(cfg, poller)

    # Prompt ran with permission_mode="plan"
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert "design the database" in call[0]
    assert len(runner.recorded_options) == 1
    assert runner.recorded_options[0].permission_mode == "plan"
    # Never steered into running turns
    assert len(steer_calls) == 0


@pytest.mark.anyio
async def test_loop_plan_prompt_unsupported_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    async def poller(_cfg: TelegramBridgeConfig):
        yield _msg("/plan design the database")

    await telegram_loop.run_main_loop(cfg, poller)

    # No runner calls dispatched
    assert len(runner.calls) == 0
    # Telegram received error notice
    sent_text = transport.send_calls[-1]["message"].text if transport.send_calls else ""
    assert "Plan mode is only available for Antigravity and Claude Code." in sent_text
    assert "<b>codex</b>" in sent_text


@pytest.mark.anyio
async def test_loop_plan_directive_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "untether.toml"
    transport = FakeTransport()
    runner = RecordingRunner([Return(answer="Plan created")], engine="antigravity")
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

    async def poller(_cfg: TelegramBridgeConfig):
        yield _msg("/antigravity /plan build microservice")

    await telegram_loop.run_main_loop(cfg, poller)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert "build microservice" in call[0]
    assert len(runner.recorded_options) == 1
    assert runner.recorded_options[0].permission_mode == "plan"


def test_coalesce_plan_mode_barrier() -> None:
    msg1 = _msg("text 1")
    msg2 = _msg("text 2")
    p1 = telegram_loop._PendingPrompt(
        msg=msg1,
        text="text 1",
        ambient_context=None,
        chat_project=None,
        topic_key=None,
        chat_session_key=None,
        reply_ref=None,
        reply_id=None,
        is_voice_transcribed=False,
        forwards=[],
        permission_mode_override=None,
    )
    p2 = telegram_loop._PendingPrompt(
        msg=msg2,
        text="text 2",
        ambient_context=None,
        chat_project=None,
        topic_key=None,
        chat_session_key=None,
        reply_ref=None,
        reply_id=None,
        is_voice_transcribed=False,
        forwards=[],
        permission_mode_override="plan",
    )
    assert _merge_block_reason(p1, p2) == "permission_mode"


@pytest.mark.anyio
async def test_action_plan_backwards_compatibility() -> None:
    executed = False

    async def sample_action():
        nonlocal executed
        executed = True

    replied_text = None

    async def fake_reply(*, text: str):
        nonlocal replied_text
        replied_text = text

    plan = ActionPlan(reply_text="done", actions=(sample_action,))
    await plan.execute(fake_reply)
    assert executed is True
    assert replied_text == "done"
