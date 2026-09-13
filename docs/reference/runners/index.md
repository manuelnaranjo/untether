# Runners

Runner docs describe the **engine-specific** behaviour: event shapes, JSON streaming, and integration notes.

Interactive features (approval buttons, plan mode, AskUserQuestion, live sessions, steer) are **Claude Code-only**; Codex, OpenCode, Pi, and Antigravity run non-interactively.

- Claude Code: [Runner](claude/runner.md), [Stream JSON cheatsheet](claude/stream-json-cheatsheet.md), [Untether events](claude/untether-events.md)
- Codex: [Exec JSON cheatsheet](codex/exec-json-cheatsheet.md), [Untether events](codex/untether-events.md)
- OpenCode: [Runner](opencode/runner.md), [Stream JSON cheatsheet](opencode/stream-json-cheatsheet.md), [Untether events](opencode/untether-events.md)
- Pi: [Runner](pi/runner.md), [Stream JSON cheatsheet](pi/stream-json-cheatsheet.md), [Untether events](pi/untether-events.md)
- Antigravity: [Runner](antigravity/runner.md), [Stream JSON cheatsheet](antigravity/stream-json-cheatsheet.md), [Untether events](antigravity/untether-events.md)

## ⚠️ Deprecated runners

Deprecated in **v0.35.5** and targeted for **removal in 0.36.0**.
It still loads but is unsupported: security and doc-accuracy fixes only, no
feature work, excluded from the release test matrix, and when a cross-engine
sweep breaks one the test is `xfail`/`skip`ped rather than the runner fixed.

- AMP (⚠️ deprecated — integration unmaintained, [#458](https://github.com/littlebearapps/untether/issues/458)): [Runner](amp/runner.md), [Stream JSON cheatsheet](amp/stream-json-cheatsheet.md), [Untether events](amp/untether-events.md)

