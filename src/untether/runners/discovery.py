"""Dynamic model and effort level discovery across supported engines."""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Sequence
from typing import Any

from ..model import DiscoveredModel

_MODEL_CACHE_TTL = 300.0  # 5 minutes
_models_cache: dict[str, tuple[float, list[DiscoveredModel]]] = {}


def get_cached_models(engine: str) -> list[DiscoveredModel]:
    """Retrieve cached models for an engine if available."""
    cached = _models_cache.get(engine)
    return cached[1] if cached is not None else []


def set_cached_models(
    models: list[DiscoveredModel],
    engine: str = "antigravity",
    timestamp: float | None = None,
) -> None:
    """Explicitly cache models for an engine."""
    now = time.monotonic() if timestamp is None else timestamp
    _models_cache[engine] = (now, models)


def reset_model_cache(engine: str | None = None) -> None:
    """Clear cached models for one or all engines."""
    global _models_cache
    if engine is None:
        _models_cache.clear()
    else:
        _models_cache.pop(engine, None)


async def fetch_available_models(
    engine: str,
    runner: Any = None,
    *,
    force_refresh: bool = False,
    **kwargs: Any,
) -> list[DiscoveredModel]:
    """Fetch available models for an engine lazily with TTL caching."""
    now = time.monotonic()
    if not force_refresh:
        cached = _models_cache.get(engine)
        if cached is not None and now - cached[0] < _MODEL_CACHE_TTL:
            return cached[1]

    models: list[DiscoveredModel] = []
    if runner is not None and hasattr(runner, "fetch_available_models"):
        with contextlib.suppress(Exception):
            models = await runner.fetch_available_models()

    if not models:
        if engine == "antigravity":
            from .antigravity import fetch_antigravity_models

            cmd = kwargs.get("antigravity_cmd")
            models = await fetch_antigravity_models(antigravity_cmd=cmd)
        elif engine == "codex":
            from .codex import fetch_codex_models

            cmd = kwargs.get("codex_cmd") or (
                runner.command() if runner and hasattr(runner, "command") else "codex"
            )
            models = await fetch_codex_models(codex_cmd=cmd)
        elif engine == "claude":
            from .claude import fetch_claude_models

            cmd = kwargs.get("claude_cmd") or (
                runner.command() if runner and hasattr(runner, "command") else "claude"
            )
            models = await fetch_claude_models(claude_cmd=cmd)
        elif engine == "opencode":
            from .opencode import fetch_opencode_models

            cmd = kwargs.get("opencode_cmd") or (
                runner.command()
                if runner and hasattr(runner, "command")
                else "opencode"
            )
            models = await fetch_opencode_models(opencode_cmd=cmd)
        elif engine == "pi":
            from .pi import fetch_pi_models

            cmd = kwargs.get("pi_cmd") or (
                runner.command() if runner and hasattr(runner, "command") else "pi"
            )
            models = await fetch_pi_models(pi_cmd=cmd)

    _models_cache[engine] = (now, models)
    return models


async def get_model_effort_levels(
    engine: str,
    model_id: str | None,
    runner: Any = None,
    *,
    antigravity_cmd: str | None = None,
    **kwargs: Any,
) -> tuple[str, ...]:
    """Resolve effort/thinking levels available for a model on a given engine."""
    if not model_id:
        return ()

    models = await fetch_available_models(
        engine, runner=runner, antigravity_cmd=antigravity_cmd, **kwargs
    )
    match = re.match(r"^(.*?)-(low|medium|high)$", model_id)
    base_id = match.group(1) if match else model_id
    for m in models:
        m_match = re.match(r"^(.*?)-(low|medium|high)$", m.model_id)
        m_base = m_match.group(1) if m_match else m.model_id
        if (
            m.model_id == model_id
            or m.model_id == base_id
            or m_base == base_id
            or model_id in m.raw_ids
            or base_id in m.raw_ids
        ):
            return m.effort_levels

    if engine == "antigravity":
        from .antigravity import antigravity_model_supports_effort

        if antigravity_model_supports_effort(model_id, discovered_models=models):
            return ("low", "medium", "high")
        return ()

    from ..telegram.engine_overrides import allowed_reasoning_levels, supports_reasoning

    if supports_reasoning(engine):
        return allowed_reasoning_levels(engine)
    return ()


def model_supports_effort(
    engine: str,
    model: str | None,
    discovered_models: Sequence[DiscoveredModel] | None = None,
) -> bool:
    """Check whether a model on the given engine supports effort/reasoning levels."""
    if not model:
        return True

    from ..telegram.engine_overrides import supports_reasoning

    if not supports_reasoning(engine):
        return False

    if engine == "antigravity":
        from .antigravity import antigravity_model_supports_effort

        return antigravity_model_supports_effort(
            model, discovered_models=discovered_models
        )

    models = (
        discovered_models
        if discovered_models is not None
        else get_cached_models(engine)
    )
    if models:
        m_str = model.strip()
        match = re.match(r"^(.*?)-(low|medium|high)$", m_str)
        base = match.group(1) if match else m_str
        for m in models:
            m_id = m.model_id.lower()
            raw_ids = [r.lower() for r in m.raw_ids]
            if (
                m_id == m_str.lower()
                or m_id == base.lower()
                or m_str.lower() in raw_ids
                or base.lower() in raw_ids
            ):
                return m.supports_effort

    return True
