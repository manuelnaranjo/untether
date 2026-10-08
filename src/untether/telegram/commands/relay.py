"""Command backend for relaying slash commands directly to the engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...commands import CommandBackend, CommandContext, CommandResult
from ...directives import unescape_relayed_command

if TYPE_CHECKING:
    pass

__all__ = [
    "BACKEND",
    "RelayCommand",
    "split_relay_command",
    "unescape_relayed_command",
]


def split_relay_command(command_id: str | None, args_text: str | None) -> str | None:
    """``/relay <command>`` -> text of command; None otherwise (bare ``/relay``
    which opens the help message, or other commands)."""
    if command_id != "relay":
        return None
    text = (args_text or "").strip()
    if not text:
        return None
    if not text.startswith("/"):
        text = f"/{text}"
    return text


class RelayCommand:
    """Relay a slash command directly to the active engine."""

    id = "relay"
    description = "Relay a slash command directly to the engine"

    async def handle(self, ctx: CommandContext) -> CommandResult | None:
        return CommandResult(
            text=(
                "**Relay an engine command directly:**\n\n"
                "• `/relay <command>` — send a slash command to the engine (e.g. `/relay /compact`)\n"
                "• `//<command>` — shorthand escape (e.g. `//compact`, `//model`)\n\n"
                "Untether will pass the command straight to the active engine without intercepting it."
            ),
            notify=True,
        )


BACKEND: CommandBackend = RelayCommand()
