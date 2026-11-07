from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ...commands import CommandBackend, CommandContext, CommandResult, RunRequest
from ...context import RunContext
from ...logging import get_logger
from ...runners.run_options import CLAUDE_PLAN_AUTO_MODE, claude_cli_permission_mode
from ._permission_mode_text import (
    APPLY_TIMING_TEXT,
    NO_OVERRIDE_LABEL,
    NO_OVERRIDE_TEXT,
    cli_name_suffix,
    mode_display,
)
from .overrides import SETTINGS_ADMIN_GATE, AdminGate

logger = get_logger(__name__)

PLAN_USAGE = (
    "usage: `/plan <task>`, `/plan`, `/plan on`, `/plan off`,"
    " `/plan show`, or `/plan clear`"
)

_PLAN_SETTING_SUBCOMMANDS = frozenset(
    {
        "on",
        "off",
        "show",
        "clear",
        "plan",
        "plan-auto",
        "auto",
    }
)

PERMISSION_MODES = {
    "on": "plan",
    "plan-auto": CLAUDE_PLAN_AUTO_MODE,
    "auto": "auto",
    "off": "acceptEdits",
}

_PLANNING_MODES = ("plan", CLAUDE_PLAN_AUTO_MODE)

_SUPPORTED_ENGINES = frozenset({"antigravity", "claude"})

_AGY_PLAN_ONLY = (
    "<b>{arg}</b> isn't available for Antigravity — use /plan on / off,"
    " or /config → Permission mode."
)


@dataclass(frozen=True, slots=True)
class ActionPlan:
    reply_text: str | None
    actions: tuple[Callable[[], Awaitable[None]], ...] = ()

    async def execute(self, reply: Callable[..., Awaitable[None]]) -> None:
        for action in self.actions:
            await action()
        if self.reply_text is not None:
            await reply(text=self.reply_text)


def split_plan_command(
    command_id: str | None, args_text: str | None
) -> tuple[str, str] | None:
    """``/plan <task>`` → ("plan", task); None otherwise (settings subcommands or bare toggle)."""
    if command_id != "plan":
        return None
    text = (args_text or "").strip()
    if not text:
        return None
    words = text.split(None, 1)
    if len(words) == 1 and words[0].lower() in _PLAN_SETTING_SUBCOMMANDS:
        return None
    return "plan", text


def _summary(mode: str) -> str:
    _, mode_text = mode_display(mode)
    return mode_text.summary if mode_text is not None else mode


class PlanCommand:
    """Command backend for planning a specific task or toggling plan mode."""

    id = "plan"
    description = "Plan a task or toggle Plan first mode (on/off/show/clear)"

    @staticmethod
    def admin_gate(args_text: str) -> AdminGate | None:
        """#1042: changing the chat's permission mode needs a group admin.

        Prompts (/plan <task>) and 'show' remain open to anyone.
        """
        split = split_plan_command("plan", args_text)
        if split is not None:
            return None
        args = args_text.strip().lower()
        if args == "" or args == "clear" or args in {"on", "off", "auto", "plan-auto"}:
            return SETTINGS_ADMIN_GATE
        return None

    async def handle(self, ctx: CommandContext) -> CommandResult | None:
        from ._resolve_engine import resolve_effective_engine

        config_path = ctx.config_path
        if config_path is None:
            return CommandResult(
                text="permission mode overrides unavailable (no config path).",
                notify=True,
            )

        current_engine = await resolve_effective_engine(ctx)
        if current_engine not in _SUPPORTED_ENGINES:
            hint = ""
            if current_engine in {"codex", "gemini"}:
                hint = " Use /config → Approval policy instead."
            return CommandResult(
                text=(
                    f"Plan mode is only available for Antigravity and Claude Code."
                    f" Current engine: <b>{current_engine}</b>.{hint}"
                ),
                notify=True,
                parse_mode="HTML",
            )

        split = split_plan_command(self.id, ctx.args_text)
        if split is not None:
            prompt_text = split[1]
            if ctx.executor is not None:
                await ctx.executor.run_one(
                    RunRequest(
                        prompt=prompt_text,
                        engine=current_engine,
                        context=RunContext(permission_mode="plan"),
                    )
                )
                return None
            return CommandResult(text=f"Planning: {prompt_text}")

        if current_engine == "antigravity":
            return await _handle_antigravity(ctx, config_path)
        return await _handle_claude(ctx, config_path, current_engine)


async def _handle_antigravity(ctx: CommandContext, config_path) -> CommandResult:
    from ..chat_prefs import ChatPrefsStore, resolve_prefs_path
    from ..engine_overrides import with_override
    from . import _antigravity_mode_text as agy_mode

    prefs = ChatPrefsStore(resolve_prefs_path(config_path))
    chat_id = ctx.message.channel_id
    engine = agy_mode.ENGINE
    args = ctx.args_text.strip().lower()

    async def _effective() -> str:
        value, _source = await agy_mode.antigravity_effective_mode(
            prefs, chat_id, ctx.runtime
        )
        return agy_mode.canonical_mode(value)

    before = await _effective()

    async def _describe(prefix: str) -> CommandResult:
        value, source = await agy_mode.antigravity_effective_mode(
            prefs, chat_id, ctx.runtime
        )
        mode = agy_mode.ANTIGRAVITY_MODE_TEXT[agy_mode.canonical_mode(value)]
        if agy_mode.needs_new(before, value):
            prefix = prefix.replace(
                "applies from your next message", agy_mode.NEW_NEEDED_TEXT
            ).replace(
                "override cleared", f"override cleared ({agy_mode.NEW_NEEDED_TEXT})"
            )
        return CommandResult(
            text=(
                f"{prefix}<b>{mode.label}</b>{agy_mode.source_suffix(source)}:"
                f" {mode.hint}."
            ),
            notify=True,
            parse_mode="HTML",
        )

    async def _store(value: str | None) -> None:
        current = await prefs.get_engine_override(chat_id, engine)
        await prefs.set_engine_override(
            chat_id, engine, with_override(current, permission_mode=value)
        )

    if args == "show":
        return await _describe("permission mode: ")
    if args in ("auto", "plan-auto"):
        return CommandResult(
            text=_AGY_PLAN_ONLY.format(arg=args), notify=True, parse_mode="HTML"
        )
    if args == "":
        value, _source = await agy_mode.antigravity_effective_mode(
            prefs, chat_id, ctx.runtime
        )
        args = "off" if agy_mode.canonical_mode(value) == "plan" else "on"
    if args == "on":
        await _store("plan")
        logger.info(
            "plan.set",
            chat_id=chat_id,
            sender_id=ctx.message.sender_id,
            mode="on",
            engine=engine,
        )
        return await _describe(
            "applies from your next message. permission mode for this chat: "
        )
    if args == "off":
        await _store(None)
        value, _source = await agy_mode.antigravity_effective_mode(
            prefs, chat_id, ctx.runtime
        )
        if agy_mode.canonical_mode(value) == "plan":
            await _store("workspace")
        logger.info(
            "plan.set",
            chat_id=chat_id,
            sender_id=ctx.message.sender_id,
            mode="off",
            engine=engine,
        )
        return await _describe(
            "applies from your next message. permission mode for this chat: "
        )
    if args == "clear":
        await _store(None)
        logger.info(
            "plan.cleared",
            chat_id=chat_id,
            sender_id=ctx.message.sender_id,
            engine=engine,
        )
        return await _describe("override cleared. permission mode: ")
    return CommandResult(text=PLAN_USAGE, notify=True)


async def _handle_claude(
    ctx: CommandContext, config_path, engine: str
) -> CommandResult:
    from ..chat_prefs import ChatPrefsStore, resolve_prefs_path
    from ..engine_overrides import with_override

    chat_prefs = ChatPrefsStore(resolve_prefs_path(config_path))
    chat_id = ctx.message.channel_id
    args = ctx.args_text.strip().lower()

    if args == "show":
        current = await chat_prefs.get_engine_override(chat_id, engine)
        mode = current.permission_mode if current else None
        if mode is None:
            text = (
                f"permission mode: <b>{NO_OVERRIDE_LABEL}</b>: no override"
                f" for this chat; it {NO_OVERRIDE_TEXT}."
            )
        else:
            ui_name, mode_text = mode_display(mode)
            text = f"permission mode: <b>{ui_name}</b>{cli_name_suffix(mode)}"
            if mode_text is not None:
                text += f": {mode_text.summary}."
        return CommandResult(text=text, notify=True, parse_mode="HTML")

    if args == "":
        current = await chat_prefs.get_engine_override(chat_id, engine)
        current_mode = current.permission_mode if current else None
        args = "off" if current_mode in _PLANNING_MODES else "on"

    if args in PERMISSION_MODES:
        mode = PERMISSION_MODES[args]
        current = await chat_prefs.get_engine_override(chat_id, engine)
        updated = with_override(current, permission_mode=mode)
        await chat_prefs.set_engine_override(chat_id, engine, updated)
        cli_mode = claude_cli_permission_mode(mode)
        logger.info(
            "plan.set",
            chat_id=chat_id,
            sender_id=ctx.message.sender_id,
            mode=args,
            cli_mode=cli_mode,
            command="plan",
        )
        return CommandResult(
            text=(
                f"permission mode <b>{args}</b> for this chat:"
                f" {_summary(mode)}.\n"
                "applies from your next message"
                f" (<code>--permission-mode {cli_mode}</code>)."
                f" {APPLY_TIMING_TEXT}"
            ),
            notify=True,
            parse_mode="HTML",
        )

    if args == "clear":
        current = await chat_prefs.get_engine_override(chat_id, engine)
        updated = with_override(current, permission_mode=None)
        await chat_prefs.set_engine_override(chat_id, engine, updated)
        logger.info(
            "plan.cleared",
            chat_id=chat_id,
            sender_id=ctx.message.sender_id,
            command="plan",
        )
        return CommandResult(
            text=(
                "permission mode <b>override cleared</b>: no override for"
                f" this chat; it {NO_OVERRIDE_TEXT}."
            ),
            notify=True,
            parse_mode="HTML",
        )

    return CommandResult(text=PLAN_USAGE, notify=True)


BACKEND: CommandBackend = PlanCommand()
