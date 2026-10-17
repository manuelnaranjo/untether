import asyncio
import contextlib
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...context import RunContext
from ...logging import get_logger
from ...model import DiscoveredModel
from ...runners.antigravity import (
    antigravity_model_supports_effort,
    default_antigravity_cmd,
    fetch_antigravity_models,
    get_last_antigravity_cmd,
    parse_and_group_antigravity_models,
    set_last_antigravity_cmd,
)
from ...runners.discovery import (
    fetch_available_models,
    get_cached_models,
    get_model_effort_levels,
    model_supports_effort,
    set_cached_models,
)
from ...runners.discovery import (
    reset_model_cache as runner_reset_model_cache,
)
from ...transport import MessageRef, RenderedMessage
from ...utils.engine_argv import model_problem, safe_preview
from ...utils.error_display import redact_url_secrets
from ..bridge import CLEAR_MARKUP, MarkdownParts, prepare_telegram
from ..chat_prefs import ChatPrefsStore
from ..engine_overrides import (
    resolve_override_value,
    supports_reasoning,
    with_override,
)
from ..files import split_command_args
from ..topic_state import TopicStateStore
from ..topics import _topic_key
from ..types import TelegramCallbackQuery, TelegramIncomingMessage
from .overrides import (
    ENGINE_SOURCE_LABELS,
    OVERRIDE_SOURCE_LABELS,
    apply_engine_override,
    parse_set_args,
    require_admin_or_private,
    resolve_engine_selection,
)
from .reply import make_reply

if TYPE_CHECKING:
    from ..bridge import TelegramBridgeConfig

logger = get_logger(__name__)

MODEL_USAGE = (
    "usage: `/model`, `/model list [engine]`, `/model set <model>`, "
    "`/model set <engine> <model>`, or `/model clear [engine]`"
)

__all__ = [
    "DiscoveredModel",
    "ModelSelectorState",
    "ModelSelectorStateMachine",
    "ModelSelectorStep",
    "antigravity_model_supports_effort",
    "default_antigravity_cmd",
    "execute_antigravity_model_switch",
    "fetch_antigravity_models",
    "fetch_available_models",
    "get_cached_models",
    "get_last_antigravity_cmd",
    "get_model_effort_levels",
    "model_selector_state_machine",
    "model_supports_effort",
    "parse_and_group_antigravity_models",
    "reset_model_cache",
    "reset_model_selector_state",
    "resolve_antigravity_cmd",
    "resolve_antigravity_conversation_id",
    "resolve_engine_runner",
    "set_cached_models",
    "set_last_antigravity_cmd",
]


class ModelSelectorStep(str, Enum):
    SELECT_MODEL = "select_model"
    SELECT_EFFORT = "select_effort"


@dataclass(slots=True)
class ModelSelectorState:
    chat_id: int
    message_id: int
    engine: str
    selected_model: str
    effort_levels: tuple[str, ...]
    sender_id: int | None = None
    step: ModelSelectorStep = ModelSelectorStep.SELECT_EFFORT
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl: float = 600.0) -> bool:
        return time.monotonic() - self.created_at > ttl


class ModelSelectorStateMachine:
    """State machine for multi-step model selection (e.g. Antigravity models with effort levels)."""

    def __init__(self, ttl: float = 600.0) -> None:
        self._states: dict[tuple[int, int], ModelSelectorState] = {}
        self._ttl = ttl

    def get_pending(
        self,
        chat_id: int,
        message_id: int | None = None,
    ) -> ModelSelectorState | None:
        self._cleanup()
        if message_id is not None:
            state = self._states.get((chat_id, message_id))
            if state is not None:
                if state.is_expired(self._ttl):
                    self._states.pop((chat_id, message_id), None)
                    return None
                return state
        matching = [
            (k, v)
            for k, v in self._states.items()
            if k[0] == chat_id and not v.is_expired(self._ttl)
        ]
        if len(matching) == 1:
            return matching[0][1]
        return None

    def set_selecting_effort(
        self,
        *,
        chat_id: int,
        message_id: int,
        engine: str,
        model_id: str,
        effort_levels: tuple[str, ...],
        sender_id: int | None = None,
    ) -> ModelSelectorState:
        self._cleanup()
        state = ModelSelectorState(
            chat_id=chat_id,
            message_id=message_id,
            engine=engine,
            selected_model=model_id,
            effort_levels=effort_levels,
            sender_id=sender_id,
            step=ModelSelectorStep.SELECT_EFFORT,
        )
        self._states[(chat_id, message_id)] = state
        return state

    def clear(
        self,
        chat_id: int,
        message_id: int | None = None,
    ) -> ModelSelectorState | None:
        if message_id is not None and (chat_id, message_id) in self._states:
            return self._states.pop((chat_id, message_id), None)
        matching = [k for k in self._states if k[0] == chat_id]
        if matching:
            return self._states.pop(matching[0], None)
        return None

    def reset(self) -> None:
        self._states.clear()

    def _cleanup(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._states.items() if now - v.created_at > self._ttl]
        for k in expired:
            self._states.pop(k, None)


model_selector_state_machine = ModelSelectorStateMachine()


def reset_model_selector_state() -> None:
    model_selector_state_machine.reset()


def reset_model_cache(engine: str | None = None) -> None:
    runner_reset_model_cache(engine)
    model_selector_state_machine.reset()


def resolve_engine_runner(cfg: TelegramBridgeConfig | None, engine: str) -> Any:
    """Resolve runner instance for engine from config runtime if available."""
    if cfg is None or not hasattr(cfg, "runtime") or cfg.runtime is None:
        return None
    try:
        resolved = cfg.runtime.resolve_runner(resume_token=None, engine_override=engine)
        return resolved.runner if resolved is not None and resolved.available else None
    except Exception:  # noqa: BLE001
        return None


def resolve_antigravity_cmd(
    *,
    chat_prefs: ChatPrefsStore | None = None,
    antigravity_cmd: str | None = None,
) -> str:
    if antigravity_cmd:
        return antigravity_cmd
    last_cmd = get_last_antigravity_cmd()
    if last_cmd:
        return last_cmd
    if chat_prefs is not None and hasattr(chat_prefs, "_path"):
        with contextlib.suppress(Exception):
            config_path = chat_prefs._path.with_name("untether.toml")
            if config_path.exists():
                from ...config import load_config

                cfg = load_config(config_path)
                raw = cfg.get("antigravity", {}).get("cmd") or cfg.get(
                    "antigravity", {}
                ).get("antigravity_cmd")
                if raw and isinstance(raw, str):
                    import os

                    return os.path.expanduser(raw)
    return default_antigravity_cmd()


async def resolve_antigravity_conversation_id(
    chat_id: int,
    *,
    tkey: tuple[int, int] | None = None,
    topic_store: TopicStateStore | None = None,
    chat_prefs: ChatPrefsStore | None = None,
) -> str | None:
    if tkey is not None and topic_store is not None:
        with contextlib.suppress(Exception):
            token = await topic_store.get_session_resume(
                tkey[0], tkey[1], "antigravity"
            )
            if token and token.value:
                return token.value
    if chat_prefs is not None and hasattr(chat_prefs, "_path"):
        with contextlib.suppress(Exception):
            from ..chat_sessions import ChatSessionStore, resolve_sessions_path

            sessions_path = resolve_sessions_path(chat_prefs._path)
            if sessions_path.exists():
                chat_store = ChatSessionStore(sessions_path)
                token = await chat_store.get_session_resume(
                    chat_id, owner_id=None, engine="antigravity"
                )
                if token and token.value:
                    return token.value
    return None


async def execute_antigravity_model_switch(
    model: str,
    effort: str | None = None,
    *,
    conversation_id: str | None = None,
    antigravity_cmd: str | None = None,
    timeout_seconds: float = 10.0,
) -> str | None:
    """Invoke `agy` to switch/verify model and capture output."""
    cmd = resolve_antigravity_cmd(antigravity_cmd=antigravity_cmd)
    expanded = Path(cmd).expanduser()
    if not shutil.which(str(expanded)) and not expanded.exists():
        return None

    if effort:
        models = await fetch_available_models(
            "antigravity", antigravity_cmd=antigravity_cmd
        )
        if not antigravity_model_supports_effort(model, discovered_models=models):
            effort = None

    args = [str(expanded)]
    if conversation_id:
        args.extend(["--conversation", conversation_id])
    model_arg = (
        f"{model}-{effort}" if effort and not model.endswith(f"-{effort}") else model
    )
    args.extend(["--model", model_arg, "-p", "/model"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
        if proc.returncode == 0:
            out = stdout.decode("utf-8", errors="replace").strip()
            return out if out else None
        err = stderr.decode("utf-8", errors="replace").strip()
        logger.warning(
            "antigravity.model_switch.failed", error=err, code=proc.returncode
        )
        return err if err else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("antigravity.model_switch.error", error=str(exc))
        return None


async def _render_model_view(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    tkey: tuple[int, int] | None,
    *,
    explicit_engine: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    if explicit_engine is not None:
        engine = explicit_engine
        engine_source = "explicit"
    else:
        selection = await resolve_engine_selection(
            cfg,
            msg,
            ambient_context=ambient_context,
            topic_store=topic_store,
            chat_prefs=chat_prefs,
            topic_key=tkey,
        )
        if selection is None:
            return "failed to resolve engine.", None
        engine, engine_source = selection

    topic_override = None
    if tkey is not None and topic_store is not None:
        topic_override = await topic_store.get_engine_override(tkey[0], tkey[1], engine)
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(msg.chat_id, engine)
    resolution = resolve_override_value(
        topic_override=topic_override,
        chat_override=chat_override,
        field="model",
    )
    source_label = ENGINE_SOURCE_LABELS.get(engine_source, engine_source)
    engine_line = f"engine: {engine} ({source_label})"
    model_value = resolution.value or "default"
    model_line = (
        f"model: **{model_value}** ({OVERRIDE_SOURCE_LABELS[resolution.source]})"
    )
    topic_label = resolution.topic_value or "none"
    if tkey is None:
        topic_label = "none"
    chat_label = (
        "unavailable" if chat_prefs is None else resolution.chat_value or "none"
    )
    defaults_line = f"defaults: topic: {topic_label}, chat: {chat_label}"
    available_line = f"available engines: {', '.join(cfg.runtime.engine_ids)}"
    sections = [engine_line, model_line, defaults_line, available_line]

    runner = resolve_engine_runner(cfg, engine)
    models = await fetch_available_models(engine, runner=runner)
    if not models:
        return "\n\n".join(sections), None

    model_lines = [f"available models ({engine}):"]
    buttons: list[list[dict[str, str]]] = []
    for m in models:
        active = False
        if resolution.value:
            res_val = resolution.value.lower()
            m_id = m.model_id.lower()
            if res_val == m_id or res_val in [r.lower() for r in m.raw_ids]:
                active = True
            else:
                m_match = re.match(r"^(.*?)-(low|medium|high)$", res_val)
                if m_match and m_match.group(1) == m_id:
                    active = True
        check = "✓ " if active else ""
        if m.supports_effort:
            efforts_str = ", ".join(m.effort_levels)
            model_lines.append(f"• {m.model_id} (effort: {efforts_str})")
        else:
            model_lines.append(f"• {m.model_id}")
        buttons.append(
            [
                {
                    "text": f"{check}{m.model_id}",
                    "callback_data": f"model:set:{m.model_id}",
                }
            ]
        )

    action_row = [{"text": "Clear model", "callback_data": "model:clear"}]
    action_row.append({"text": "🔄 Refresh", "callback_data": "model:refresh"})
    buttons.append(action_row)

    sections.append("\n".join(model_lines))
    return "\n\n".join(sections), {"inline_keyboard": buttons}


async def _render_effort_view(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    *,
    engine: str,
    model_id: str,
    effort_levels: tuple[str, ...],
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    tkey: tuple[int, int] | None,
) -> tuple[str, dict[str, Any]]:
    topic_override = None
    if tkey is not None and topic_store is not None:
        topic_override = await topic_store.get_engine_override(tkey[0], tkey[1], engine)
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(msg.chat_id, engine)
    model_res = resolve_override_value(
        topic_override=topic_override,
        chat_override=chat_override,
        field="model",
    )
    resolution = resolve_override_value(
        topic_override=topic_override,
        chat_override=chat_override,
        field="reasoning",
    )
    res_model = (model_res.value or "").lower()
    model_matches = res_model == model_id.lower() or res_model.startswith(
        f"{model_id.lower()}-"
    )
    current_effort = (resolution.value or "").lower() if model_matches else ""

    text_parts = [
        f"engine: {engine}",
        f"model: **{model_id}**",
        f"Select effort for `{model_id}`:",
    ]

    effort_buttons: list[dict[str, str]] = []
    for lvl in effort_levels:
        check = "✓ " if current_effort == lvl.lower() else ""
        effort_buttons.append(
            {"text": f"{check}{lvl.capitalize()}", "callback_data": f"effort:set:{lvl}"}
        )

    action_buttons = [
        {"text": "Clear effort", "callback_data": "effort:clear"},
        {"text": "« Back to models", "callback_data": "model:back"},
    ]

    keyboard = {"inline_keyboard": [effort_buttons, action_buttons]}
    return "\n\n".join(text_parts), keyboard


async def _remaining_model(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    engine: str,
    *,
    tkey: tuple[int, int] | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
) -> tuple[str, str] | None:
    """``(source label, model)`` Untether passes to *engine* once an override
    is cleared — a remaining override, else the engine's configured model —
    or None when it passes no model at all (#998)."""
    topic_override = None
    if tkey is not None and topic_store is not None:
        topic_override = await topic_store.get_engine_override(tkey[0], tkey[1], engine)
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(msg.chat_id, engine)
    resolution = resolve_override_value(
        topic_override=topic_override, chat_override=chat_override, field="model"
    )
    if resolution.value:
        label = (
            "topic override"
            if resolution.source == "topic_override"
            else "chat override"
        )
        return (label, resolution.value)
    try:
        runner = cfg.runtime.resolve_runner(
            resume_token=None, engine_override=engine
        ).runner
    except Exception:  # noqa: BLE001 — a reply line must never break /model
        return None
    configured = getattr(runner, "model", None)
    if isinstance(configured, str) and configured:
        return ("configured default", configured)
    return None


# `/model set` and `/model clear` (#1155) offer the same way to a clean start:
# a resumed session can keep the model it was using (Pi stores it in its
# session file).
_NEW_SESSION_HINT = "If you want a clean start on the new model, run `/new`."


def _after_clear_line(engine: str, remaining: tuple[str, str] | None) -> str:
    """#998: what the next run gets. Every runner passes a model it knows on
    resumed runs too; with none, the engine decides and a resumed session can
    keep the cleared model (Claude Code 2.1.296 and Pi 0.78.0 do), so the
    only sure way back to the engine's default is a fresh session."""
    if remaining is not None:
        label, model = remaining
        return (
            f"From the next run Untether passes the {label} "
            f"`{redact_url_secrets(model)}`.\n"
            f"{_NEW_SESSION_HINT}"
        )
    return (
        f"Untether no longer chooses a model for `{engine}`, and a resumed "
        "session may stay on the model it was using. Run `/new` to start a "
        "fresh session on the engine's default."
    )


async def _handle_model_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
) -> None:
    reply = make_reply(cfg, msg)
    tkey = (
        _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
        if topic_store is not None
        else None
    )
    tokens = split_command_args(args_text)
    action = tokens[0].lower() if tokens else "show"
    engine_ids = {engine.lower() for engine in cfg.runtime.engine_ids}

    if action in {"show", "", "list"}:
        explicit_engine = None
        if action == "list" and len(tokens) >= 2:
            arg = tokens[1].strip().lower()
            if arg in engine_ids:
                explicit_engine = arg
            else:
                available = ", ".join(cfg.runtime.engine_ids)
                await reply(
                    text=f"unknown engine `{arg}`.\navailable engines: `{available}`"
                )
                return

        text, reply_markup = await _render_model_view(
            cfg,
            msg,
            ambient_context=ambient_context,
            topic_store=topic_store,
            chat_prefs=chat_prefs,
            tkey=tkey,
            explicit_engine=explicit_engine,
        )
        if reply_markup is not None:
            await reply(text=text, reply_markup=reply_markup)
        else:
            await reply(text=text)
        return

    if action == "set":
        engine_arg, model = parse_set_args(tokens, engine_ids=engine_ids)
        if model is None:
            await reply(text=MODEL_USAGE)
            return
        # #1034: the value becomes part of an engine CLI's argv. Refuse one
        # that could read as a flag before it is stored (never echoed back).
        problem = model_problem(model)
        if problem is not None:
            logger.warning(
                "model.set.invalid",
                chat_id=msg.chat_id,
                preview=safe_preview(model),
                command="model",
            )
            await reply(text=f"invalid model: a model id {problem}.\n\n{MODEL_USAGE}")
            return
        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for model overrides.",
            failed_member="failed to verify model override permissions.",
            denied="changing model overrides is restricted to group admins.",
        ):
            return
        if engine_arg is None:
            selection = await resolve_engine_selection(
                cfg,
                msg,
                ambient_context=ambient_context,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                topic_key=tkey,
            )
            if selection is None:
                return
            engine, _ = selection
        else:
            engine = engine_arg
            if engine not in engine_ids:
                available = cfg.runtime.available_engines_text(code=True)
                await reply(
                    text=f"unknown engine `{engine}`.\navailable engines: {available}"
                )
                return
        runner = resolve_engine_runner(cfg, engine)
        models = await fetch_available_models(engine, runner=runner)
        supports_effort = model_supports_effort(engine, model, discovered_models=models)
        try:
            scope = await apply_engine_override(
                reply=reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=msg.chat_id,
                engine=engine,
                update=lambda current: with_override(
                    current,
                    model=model,
                    reasoning=(
                        None
                        if not supports_effort
                        else (current.reasoning if current is not None else None)
                    ),
                ),
                topic_unavailable="topic model overrides are unavailable.",
                chat_unavailable="chat model overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model.override.failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            await reply(text="failed to apply model override.")
            return
        if scope is None:
            return
        logger.info(
            "model.set",
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            engine=engine,
            model=model,
            scope=scope,
            command="model",
        )
        shown = redact_url_secrets(model)  # #1161
        await reply(
            text=(
                f"{scope} model override **set to** `{shown}` for `{engine}`.\n"
                f"{_NEW_SESSION_HINT}"
            )
        )
        return

    if action == "clear":
        engine = None
        if len(tokens) > 2:
            await reply(text=MODEL_USAGE)
            return
        if len(tokens) == 2:
            engine = tokens[1].strip().lower() or None
        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for model overrides.",
            failed_member="failed to verify model override permissions.",
            denied="changing model overrides is restricted to group admins.",
        ):
            return
        if engine is None:
            selection = await resolve_engine_selection(
                cfg,
                msg,
                ambient_context=ambient_context,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                topic_key=tkey,
            )
            if selection is None:
                return
            engine, _ = selection
        if engine not in engine_ids:
            available = cfg.runtime.available_engines_text(code=True)
            await reply(
                text=f"unknown engine `{engine}`.\navailable engines: {available}"
            )
            return
        try:
            scope = await apply_engine_override(
                reply=reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=msg.chat_id,
                engine=engine,
                update=lambda current: with_override(current, model=None),
                topic_unavailable="topic model overrides are unavailable.",
                chat_unavailable="chat model overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model.override.failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            await reply(text="failed to clear model override.")
            return
        if scope is None:
            return
        logger.info(
            "model.cleared",
            chat_id=msg.chat_id,
            sender_id=msg.sender_id,
            engine=engine,
            scope=scope,
        )
        remaining = await _remaining_model(
            cfg, msg, engine, tkey=tkey, topic_store=topic_store, chat_prefs=chat_prefs
        )
        await reply(
            text=(
                f"{scope} model override **cleared** for `{engine}`.\n"
                f"{_after_clear_line(engine, remaining)}"
            )
        )
        return

    await reply(text=MODEL_USAGE)


async def _handle_callback_model(
    cfg: TelegramBridgeConfig,
    query: TelegramCallbackQuery,
    args_text: str,
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    *,
    scope_chat_ids: frozenset[int] | None = None,
    thread_id: int | None = None,
) -> None:
    if (
        cfg.allowed_user_ids
        and query.sender_id is not None
        and query.sender_id not in cfg.allowed_user_ids
    ):
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text="Not authorised",
            )
        return

    msg = TelegramIncomingMessage(
        transport="telegram",
        chat_id=query.chat_id,
        message_id=query.message_id,
        text="",
        thread_id=thread_id,
        reply_to_message_id=None,
        reply_to_text=None,
        sender_id=query.sender_id,
    )
    tkey = (
        _topic_key(msg, cfg, scope_chat_ids=scope_chat_ids)
        if topic_store is not None
        else None
    )
    selection = await resolve_engine_selection(
        cfg,
        msg,
        ambient_context=ambient_context,
        topic_store=topic_store,
        chat_prefs=chat_prefs,
        topic_key=tkey,
    )
    if selection is None:
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text="Failed to resolve engine",
            )
        return
    engine, _ = selection

    parts = args_text.split(":", 1)
    action = parts[0].strip().lower()

    if action == "set":
        model = parts[1].strip() if len(parts) > 1 else None
        if not model:
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="No model specified",
                )
            return

        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for model overrides.",
            failed_member="failed to verify model override permissions.",
            denied="changing model overrides is restricted to group admins.",
        ):
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Restricted to group admins",
                )
            return

        runner = resolve_engine_runner(cfg, engine)
        models = await fetch_available_models(engine, runner=runner)
        matched_m = next(
            (m for m in models if m.model_id == model or model in m.raw_ids),
            None,
        )
        supports_effort = (
            matched_m.supports_effort
            if matched_m is not None
            else model_supports_effort(engine, model, discovered_models=models)
        )
        if supports_reasoning(engine) and supports_effort:
            model_id = matched_m.model_id if matched_m is not None else model
            effort_levels = (
                matched_m.effort_levels
                if matched_m is not None and matched_m.effort_levels
                else await get_model_effort_levels(engine, model_id, runner=runner)
            )
            if effort_levels:
                model_selector_state_machine.set_selecting_effort(
                    chat_id=query.chat_id,
                    message_id=query.message_id,
                    engine=engine,
                    model_id=model_id,
                    effort_levels=effort_levels,
                    sender_id=query.sender_id,
                )
                if query.callback_query_id is not None:
                    await cfg.bot.answer_callback_query(
                        callback_query_id=query.callback_query_id,
                        text=f"Model set to {model}. Choose effort:",
                    )
                effort_text, effort_kb = await _render_effort_view(
                    cfg,
                    msg,
                    engine=engine,
                    model_id=model_id,
                    effort_levels=effort_levels,
                    ambient_context=ambient_context,
                    topic_store=topic_store,
                    chat_prefs=chat_prefs,
                    tkey=tkey,
                )
                rendered_text, entities = prepare_telegram(
                    MarkdownParts(header=effort_text)
                )
                extra: dict[str, Any] = {
                    "entities": entities,
                    "reply_markup": effort_kb,
                }
                try:
                    await cfg.exec_cfg.transport.edit(
                        ref=MessageRef(
                            channel_id=query.chat_id, message_id=query.message_id
                        ),
                        message=RenderedMessage(text=rendered_text, extra=extra),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("model.callback.edit_failed", error=str(exc))
                return

        model_selector_state_machine.clear(query.chat_id, query.message_id)

        async def _noop_reply(**kwargs: Any) -> None:
            pass

        try:
            await apply_engine_override(
                reply=_noop_reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=query.chat_id,
                engine=engine,
                update=lambda current: with_override(
                    current,
                    model=model,
                    reasoning=(
                        None
                        if not supports_effort
                        else (current.reasoning if current is not None else None)
                    ),
                ),
                topic_unavailable="topic model overrides are unavailable.",
                chat_unavailable="chat model overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model.callback.override_failed", error=str(exc))
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Failed to apply model override",
                )
            return

        logger.info(
            "model.callback.set",
            chat_id=query.chat_id,
            engine=engine,
            model=model,
        )

        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text=f"Model set to {model}",
            )

    elif action == "effort":
        effort_parts = parts[1].split(":") if len(parts) > 1 else []
        pending = model_selector_state_machine.get_pending(
            query.chat_id, query.message_id
        )
        if len(effort_parts) >= 2:
            model = effort_parts[0].strip()
            level = effort_parts[1].strip().lower()
        elif len(effort_parts) == 1:
            level = effort_parts[0].strip().lower()
            model = pending.selected_model if pending else None
        else:
            level = None
            model = pending.selected_model if pending else None

        if not model or not level:
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Invalid effort selection",
                )
            return

        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for model overrides.",
            failed_member="failed to verify model override permissions.",
            denied="changing model overrides is restricted to group admins.",
        ):
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Restricted to group admins",
                )
            return

        async def _noop_reply(**kwargs: Any) -> None:
            pass

        try:
            await apply_engine_override(
                reply=_noop_reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=query.chat_id,
                engine=engine,
                update=lambda current: with_override(
                    current,
                    model=model,
                    reasoning=level,
                ),
                topic_unavailable="topic model overrides are unavailable.",
                chat_unavailable="chat model overrides are unavailable (no config path).",
            )
            model_selector_state_machine.clear(query.chat_id, query.message_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("model.callback.override_failed", error=str(exc))
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Failed to apply model override",
                )
            return

        logger.info(
            "model.callback.effort_set",
            chat_id=query.chat_id,
            engine=engine,
            model=model,
            effort=level,
        )
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text=f"Model set to {model} ({level})",
            )

        agy_response = None
        if engine == "antigravity":
            conversation_id = await resolve_antigravity_conversation_id(
                query.chat_id,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
            )
            agy_cmd = resolve_antigravity_cmd(chat_prefs=chat_prefs)
            agy_response = await execute_antigravity_model_switch(
                model,
                level,
                conversation_id=conversation_id,
                antigravity_cmd=agy_cmd,
            )
        parts = [
            f"engine: {engine}",
            f"model: **{model}**",
            f"effort: **{level}**" if level else "effort: **default**",
        ]
        if agy_response:
            parts.append(agy_response)
        text = "\n\n".join(parts)
        rendered_text, entities = prepare_telegram(MarkdownParts(header=text))
        extra: dict[str, Any] = {
            "entities": entities,
            "reply_markup": CLEAR_MARKUP,
        }
        try:
            await cfg.exec_cfg.transport.edit(
                ref=MessageRef(channel_id=query.chat_id, message_id=query.message_id),
                message=RenderedMessage(text=rendered_text, extra=extra),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model.callback.edit_failed", error=str(exc))
        return

    elif action == "clear":
        model_selector_state_machine.clear(query.chat_id, query.message_id)
        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for model overrides.",
            failed_member="failed to verify model override permissions.",
            denied="changing model overrides is restricted to group admins.",
        ):
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Restricted to group admins",
                )
            return

        async def _noop_reply(**kwargs: Any) -> None:
            pass

        try:
            await apply_engine_override(
                reply=_noop_reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=query.chat_id,
                engine=engine,
                update=lambda current: with_override(
                    current,
                    model=None,
                    reasoning=current.reasoning if current is not None else None,
                ),
                topic_unavailable="topic model overrides are unavailable.",
                chat_unavailable="chat model overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model.callback.clear_failed", error=str(exc))
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Failed to clear model override",
                )
            return

        logger.info("model.callback.cleared", chat_id=query.chat_id, engine=engine)
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text="Model override cleared",
            )

    elif action == "refresh":
        model_selector_state_machine.clear(query.chat_id, query.message_id)
        reset_model_cache(engine)
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text="Models refreshed",
            )

    elif action == "back":
        model_selector_state_machine.clear(query.chat_id, query.message_id)
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text="Models",
            )

    text, reply_markup = await _render_model_view(
        cfg,
        msg,
        ambient_context=ambient_context,
        topic_store=topic_store,
        chat_prefs=chat_prefs,
        tkey=tkey,
    )
    rendered_text, entities = prepare_telegram(MarkdownParts(header=text))
    extra: dict[str, Any] = {"entities": entities}
    if reply_markup is not None:
        extra["reply_markup"] = reply_markup
    try:
        await cfg.exec_cfg.transport.edit(
            ref=MessageRef(channel_id=query.chat_id, message_id=query.message_id),
            message=RenderedMessage(text=rendered_text, extra=extra),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("model.callback.edit_failed", error=str(exc))
