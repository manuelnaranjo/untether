from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...context import RunContext
from ...logging import get_logger
from ..chat_prefs import ChatPrefsStore
from ..engine_overrides import (
    EngineOverrides,
    allowed_reasoning_levels,
    get_reasoning_label,
    resolve_override_value,
)
from ..bridge import CLEAR_MARKUP
from ..render import MarkdownParts, prepare_telegram
from ..topic_state import TopicStateStore
from ..topics import _topic_key
from ..types import TelegramCallbackQuery, TelegramIncomingMessage
from .model import (
    _render_effort_view,
    execute_antigravity_model_switch,
    get_model_effort_levels,
    model_selector_state_machine,
    resolve_antigravity_cmd,
    resolve_antigravity_conversation_id,
)
from .overrides import (
    apply_engine_override,
    require_admin_or_private,
    resolve_engine_selection,
)
from .reasoning import _handle_reasoning_command

if TYPE_CHECKING:
    from ..bridge import TelegramBridgeConfig

logger = get_logger(__name__)

EFFORT_USAGE = (
    "usage: `/effort`, `/effort set <level>`, "
    "`/effort set <engine> <level>`, or `/effort clear [engine]`"
)


async def _handle_effort_command(
    cfg: TelegramBridgeConfig,
    msg: TelegramIncomingMessage,
    args_text: str,
    ambient_context: RunContext | None,
    topic_store: TopicStateStore | None,
    chat_prefs: ChatPrefsStore | None,
    *,
    resolved_scope: str | None = None,
    scope_chat_ids: frozenset[int] | None = None,
    invoked_as: str = "effort",
) -> None:
    await _handle_reasoning_command(
        cfg,
        msg,
        args_text,
        ambient_context,
        topic_store,
        chat_prefs,
        resolved_scope=resolved_scope,
        scope_chat_ids=scope_chat_ids,
        invoked_as=invoked_as,
    )


async def _handle_callback_effort(
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
        level = parts[1].strip().lower() if len(parts) > 1 else None
        if not level:
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="No effort level specified",
                )
            return

        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for effort overrides.",
            failed_member="failed to verify effort override permissions.",
            denied="changing effort overrides is restricted to group admins.",
        ):
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Restricted to group admins",
                )
            return

        async def _noop_reply(**kwargs: Any) -> None:
            pass

        pending = model_selector_state_machine.get_pending(
            query.chat_id, query.message_id
        )
        model_to_set = pending.selected_model if pending is not None else None
        is_from_model_selector = pending is not None and model_to_set is not None

        try:
            await apply_engine_override(
                reply=_noop_reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=query.chat_id,
                engine=engine,
                update=lambda current: EngineOverrides(
                    model=model_to_set
                    or (current.model if current is not None else None),
                    reasoning=level,
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
                ),
                topic_unavailable="topic effort overrides are unavailable.",
                chat_unavailable="chat effort overrides are unavailable (no config path).",
            )
            if pending is not None:
                model_selector_state_machine.clear(
                    query.chat_id, query.message_id
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("effort.callback.set_failed", error=str(exc))
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Failed to set effort",
                )
            return

        label = get_reasoning_label(engine)
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text=f"{label} set to {level}",
            )

        if is_from_model_selector and engine == "antigravity":
            conversation_id = await resolve_antigravity_conversation_id(
                query.chat_id,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
            )
            agy_cmd = resolve_antigravity_cmd(chat_prefs=chat_prefs)
            agy_response = await execute_antigravity_model_switch(
                model_to_set,
                level,
                conversation_id=conversation_id,
                antigravity_cmd=agy_cmd,
            )
            parts = [
                f"engine: {engine}",
                f"model: **{model_to_set}**",
                f"effort: **{level}**",
            ]
            if agy_response:
                parts.append(agy_response)
            text = "\n\n".join(parts)
            from ...transport import MessageRef, RenderedMessage

            rendered_text, entities = prepare_telegram(
                MarkdownParts(header=text)
            )
            extra: dict[str, Any] = {
                "entities": entities,
                "reply_markup": CLEAR_MARKUP,
            }
            try:
                await cfg.exec_cfg.transport.edit(
                    ref=MessageRef(
                        channel_id=query.chat_id, message_id=query.message_id
                    ),
                    message=RenderedMessage(text=rendered_text, extra=extra),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("effort.callback.edit_failed", error=str(exc))
            return

    elif action == "clear":
        if not await require_admin_or_private(
            cfg,
            msg,
            missing_sender="cannot verify sender for effort overrides.",
            failed_member="failed to verify effort override permissions.",
            denied="changing effort overrides is restricted to group admins.",
        ):
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Restricted to group admins",
                )
            return

        async def _noop_reply(**kwargs: Any) -> None:
            pass

        pending = model_selector_state_machine.get_pending(
            query.chat_id, query.message_id
        )
        model_to_set = pending.selected_model if pending is not None else None
        is_from_model_selector = pending is not None and model_to_set is not None

        try:
            await apply_engine_override(
                reply=_noop_reply,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
                chat_id=query.chat_id,
                engine=engine,
                update=lambda current: EngineOverrides(
                    model=model_to_set
                    or (current.model if current is not None else None),
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
                ),
                topic_unavailable="topic effort overrides are unavailable.",
                chat_unavailable="chat effort overrides are unavailable (no config path).",
            )
            if pending is not None:
                model_selector_state_machine.clear(
                    query.chat_id, query.message_id
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("effort.callback.clear_failed", error=str(exc))
            if query.callback_query_id is not None:
                await cfg.bot.answer_callback_query(
                    callback_query_id=query.callback_query_id,
                    text="Failed to clear effort",
                )
            return

        label = get_reasoning_label(engine)
        if query.callback_query_id is not None:
            await cfg.bot.answer_callback_query(
                callback_query_id=query.callback_query_id,
                text=f"{label} override cleared",
            )

        if is_from_model_selector and engine == "antigravity":
            conversation_id = await resolve_antigravity_conversation_id(
                query.chat_id,
                tkey=tkey,
                topic_store=topic_store,
                chat_prefs=chat_prefs,
            )
            agy_cmd = resolve_antigravity_cmd(chat_prefs=chat_prefs)
            agy_response = await execute_antigravity_model_switch(
                model_to_set,
                None,
                conversation_id=conversation_id,
                antigravity_cmd=agy_cmd,
            )
            parts = [
                f"engine: {engine}",
                f"model: **{model_to_set}**",
                "effort: **default**",
            ]
            if agy_response:
                parts.append(agy_response)
            text = "\n\n".join(parts)
            from ...transport import MessageRef, RenderedMessage

            rendered_text, entities = prepare_telegram(
                MarkdownParts(header=text)
            )
            extra: dict[str, Any] = {
                "entities": entities,
                "reply_markup": CLEAR_MARKUP,
            }
            try:
                await cfg.exec_cfg.transport.edit(
                    ref=MessageRef(
                        channel_id=query.chat_id, message_id=query.message_id
                    ),
                    message=RenderedMessage(text=rendered_text, extra=extra),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("effort.callback.edit_failed", error=str(exc))
            return

    # Re-render effort view to show updated checkmarks
    topic_override = None
    if tkey is not None and topic_store is not None:
        topic_override = await topic_store.get_engine_override(tkey[0], tkey[1], engine)
    chat_override = None
    if chat_prefs is not None:
        chat_override = await chat_prefs.get_engine_override(query.chat_id, engine)
    model_res = resolve_override_value(
        topic_override=topic_override,
        chat_override=chat_override,
        field="model",
    )
    active_model = model_res.value
    effort_levels = (
        await get_model_effort_levels(engine, active_model)
        if engine == "antigravity" and active_model
        else allowed_reasoning_levels(engine)
    )

    from ...transport import MessageRef, RenderedMessage

    text, reply_markup = await _render_effort_view(
        cfg,
        msg,
        engine=engine,
        model_id=active_model or engine,
        effort_levels=effort_levels or allowed_reasoning_levels(engine),
        ambient_context=ambient_context,
        topic_store=topic_store,
        chat_prefs=chat_prefs,
        tkey=tkey,
    )
    rendered_text, entities = prepare_telegram(MarkdownParts(header=text))
    extra: dict[str, Any] = {"entities": entities, "reply_markup": reply_markup}
    try:
        await cfg.exec_cfg.transport.edit(
            ref=MessageRef(channel_id=query.chat_id, message_id=query.message_id),
            message=RenderedMessage(text=rendered_text, extra=extra),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("effort.callback.edit_failed", error=str(exc))
