from __future__ import annotations

from typing import TYPE_CHECKING

from ...context import RunContext
from ...logging import get_logger
from ..chat_prefs import ChatPrefsStore
from ..engine_overrides import (
    EngineOverrides,
    allowed_reasoning_levels,
    get_reasoning_command,
    get_reasoning_label,
    resolve_override_value,
)
from ..files import split_command_args
from ..topic_state import TopicStateStore
from ..topics import _topic_key
from ..types import TelegramIncomingMessage
from .model import _render_effort_view, get_model_effort_levels
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


def _reasoning_usage(cmd: str = "reasoning") -> str:
    return (
        f"usage: `/{cmd}`, `/{cmd} set <level>`, "
        f"`/{cmd} set <engine> <level>`, or `/{cmd} clear [engine]`"
    )


REASONING_USAGE = _reasoning_usage("reasoning")
EFFORT_USAGE = _reasoning_usage("effort")


async def _handle_reasoning_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
    invoked_as: str | None = None,
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
    default_cmd = (
        invoked_as
        if invoked_as in {"effort", "reasoning"}
        else "effort"
        if invoked_as == "efforts"
        else "reasoning"
    )

    if action in {"show", "", "list"}:
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
        engine, engine_source = selection
        label = get_reasoning_label(engine)
        term = label.lower()

        topic_override = None
        if tkey is not None and topic_store is not None:
            topic_override = await topic_store.get_engine_override(
                tkey[0], tkey[1], engine
            )
        chat_override = None
        if chat_prefs is not None:
            chat_override = await chat_prefs.get_engine_override(msg.chat_id, engine)

        if engine == "antigravity":
            model_res = resolve_override_value(
                topic_override=topic_override,
                chat_override=chat_override,
                field="model",
            )
            active_model = model_res.value
            effort_levels = (
                await get_model_effort_levels(engine, active_model)
                if active_model
                else ()
            )
            if active_model and not effort_levels:
                await reply(
                    text=(
                        f"engine: {engine}\n\n"
                        f"model: **{active_model}**\n\n"
                        f"model `{active_model}` does not support effort levels."
                    )
                )
                return
            if not effort_levels:
                effort_levels = allowed_reasoning_levels(engine)

            model_id = active_model or "antigravity"
            text, reply_markup = await _render_effort_view(
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
            await reply(text=text, reply_markup=reply_markup)
            return

        resolution = resolve_override_value(
            topic_override=topic_override,
            chat_override=chat_override,
            field="reasoning",
        )
        engine_line = f"engine: {engine} ({ENGINE_SOURCE_LABELS[engine_source]})"
        reasoning_value = resolution.value or "default"
        reasoning_line = (
            f"{term}: **{reasoning_value}** "
            f"({OVERRIDE_SOURCE_LABELS[resolution.source]})"
        )
        topic_label = resolution.topic_value or "none"
        if tkey is None:
            topic_label = "none"
        chat_label = (
            "unavailable" if chat_prefs is None else resolution.chat_value or "none"
        )
        defaults_line = f"defaults: topic: {topic_label}, chat: {chat_label}"
        available_levels = ", ".join(allowed_reasoning_levels(engine))
        available_line = f"available levels: {available_levels}"
        await reply(
            text="\n\n".join(
                [engine_line, reasoning_line, defaults_line, available_line]
            )
        )
        return

    if action == "set":
        engine_arg, level = parse_set_args(tokens, engine_ids=engine_ids)
        if engine_arg is not None:
            engine = engine_arg
            if engine not in engine_ids:
                available = ", ".join(cfg.runtime.engine_ids)
                await reply(
                    text=f"unknown engine `{engine}`.\navailable engines: `{available}`"
                )
                return
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
                return
            engine, _ = selection

        canonical_cmd = get_reasoning_command(engine)
        label = get_reasoning_label(engine)
        term = label.lower()

        if level is None:
            await reply(text=_reasoning_usage(canonical_cmd))
            return

        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender=f"cannot verify sender for {term} overrides.",
            failed_member=f"failed to verify {term} override permissions.",
            denied=f"changing {term} overrides is restricted to group admins.",
        ):
            return

        normalized_level = level.strip().lower()

        topic_override = None
        if tkey is not None and topic_store is not None:
            topic_override = await topic_store.get_engine_override(
                tkey[0], tkey[1], engine
            )
        chat_override = None
        if chat_prefs is not None:
            chat_override = await chat_prefs.get_engine_override(msg.chat_id, engine)

        model_res = resolve_override_value(
            topic_override=topic_override,
            chat_override=chat_override,
            field="model",
        )
        active_model = model_res.value
        model_effort_levels = (
            await get_model_effort_levels(engine, active_model)
            if engine == "antigravity" and active_model
            else ()
        )
        allowed = model_effort_levels or allowed_reasoning_levels(engine)

        if normalized_level not in allowed:
            model_hint = (
                f" for `{active_model}`"
                if engine == "antigravity" and active_model
                else ""
            )
            await reply(
                text=(
                    f"unknown {term} level `{level}`{model_hint}.\n"
                    f"available levels: {', '.join(allowed)}"
                )
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
                update=lambda current: EngineOverrides(
                    model=current.model if current is not None else None,
                    reasoning=normalized_level,
                    permission_mode=current.permission_mode
                    if current is not None
                    else None,
                    ask_questions=current.ask_questions
                    if current is not None
                    else None,
                    diff_preview=current.diff_preview if current is not None else None,
                    show_api_cost=current.show_api_cost
                    if current is not None
                    else None,
                    show_subscription_usage=current.show_subscription_usage
                    if current is not None
                    else None,
                    show_resume_line=current.show_resume_line
                    if current is not None
                    else None,
                    budget_enabled=current.budget_enabled
                    if current is not None
                    else None,
                    budget_auto_cancel=current.budget_auto_cancel
                    if current is not None
                    else None,
                    loop_enabled=current.loop_enabled if current is not None else None,
                ),
                topic_unavailable=f"topic {term} overrides are unavailable.",
                chat_unavailable=f"chat {term} overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reasoning.override.failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            await reply(text=f"failed to apply {term} override.")
            return
        if scope is None:
            return
        logger.info(
            "reasoning.set",
            chat_id=msg.chat_id,
            engine=engine,
            level=normalized_level,
            scope=scope,
            command=canonical_cmd,
        )
        if scope == "topic":
            await reply(
                text=(
                    f"topic {term} override **set to** `{normalized_level}` "
                    f"for `{engine}`.\n"
                    "If you want a clean start on the new setting, run `/new`."
                )
            )
            return
        await reply(
            text=(
                f"chat {term} override **set to** `{normalized_level}` for `{engine}`.\n"
                "If you want a clean start on the new setting, run `/new`."
            )
        )
        return

    if action == "clear":
        engine = None
        if len(tokens) > 2:
            await reply(text=_reasoning_usage(default_cmd))
            return
        if len(tokens) == 2:
            engine = tokens[1].strip().lower() or None

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
            available = ", ".join(cfg.runtime.engine_ids)
            await reply(
                text=f"unknown engine `{engine}`.\navailable engines: `{available}`"
            )
            return

        canonical_cmd = get_reasoning_command(engine)
        label = get_reasoning_label(engine)
        term = label.lower()

        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender=f"cannot verify sender for {term} overrides.",
            failed_member=f"failed to verify {term} override permissions.",
            denied=f"changing {term} overrides is restricted to group admins.",
        ):
            return

        try:
            scope = await apply_engine_override(
                reply=reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=msg.chat_id,
                engine=engine,
                update=lambda current: EngineOverrides(
                    model=current.model if current is not None else None,
                    reasoning=None,
                    permission_mode=current.permission_mode
                    if current is not None
                    else None,
                    ask_questions=current.ask_questions
                    if current is not None
                    else None,
                    diff_preview=current.diff_preview if current is not None else None,
                    show_api_cost=current.show_api_cost
                    if current is not None
                    else None,
                    show_subscription_usage=current.show_subscription_usage
                    if current is not None
                    else None,
                    show_resume_line=current.show_resume_line
                    if current is not None
                    else None,
                    budget_enabled=current.budget_enabled
                    if current is not None
                    else None,
                    budget_auto_cancel=current.budget_auto_cancel
                    if current is not None
                    else None,
                    loop_enabled=current.loop_enabled if current is not None else None,
                ),
                topic_unavailable=f"topic {term} overrides are unavailable.",
                chat_unavailable=f"chat {term} overrides are unavailable (no config path).",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reasoning.override.failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            await reply(text=f"failed to clear {term} override.")
            return
        if scope is None:
            return
        logger.info(
            "reasoning.cleared", chat_id=msg.chat_id, engine=engine, scope=scope
        )
        if scope == "topic":
            await reply(text=f"topic {term} override **cleared** (using chat default).")
            return
        await reply(text=f"chat {term} override **cleared**.")
        return

    selection = await resolve_engine_selection(
        cfg,
        msg,
        ambient_context=ambient_context,
        topic_store=topic_store,
        chat_prefs=chat_prefs,
        topic_key=tkey,
    )
    cmd = get_reasoning_command(selection[0]) if selection else default_cmd
    await reply(text=_reasoning_usage(cmd))
