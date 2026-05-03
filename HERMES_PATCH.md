# Required hermes-agent patch

The autobio plugin needs **one** framework hook in hermes-agent:
`ContextEngine.on_turn_end(messages)`. Without it, engines have no
per-turn visibility into the live message list — `compress()` only
fires on threshold, and `update_from_response` runs per API call
(including mid-tool-loop iterations) with no messages.

The autobio engine uses `on_turn_end` for two features:

1. **Speculative timing.** Speculation should fire when the tail is
   growing toward threshold. That's a per-turn decision with access
   to the live list. Without the hook, speculation can only be
   scheduled from inside `compress()` itself — by definition too late.
2. **Proactive CM check-ins.** CM's check-in decision depends on
   recent turn content. Without message-list visibility, CM would
   operate on counts alone, or observations would queue and surface
   lazily later.

## The change

Two diffs against hermes-agent.

### `agent/context_engine.py` — add the ABC method

```python
def on_turn_end(self, messages: List[Dict[str, Any]]) -> None:
    """Called after each turn's final assistant message lands.

    ``messages`` is the live message list as the runner sees it
    (system + every user/assistant/tool message accumulated so far).
    Default no-op; engines override to do per-turn maintenance —
    archival passes, speculative scheduling, proactive checks, etc.

    Called inside the runner's tool-loop *exit* path, AFTER the
    response has been committed and BEFORE returning control to
    the caller. Guarded by try/except in run_agent.py so a buggy
    engine cannot crash the runner.
    """
```

### `run_agent.py` — call it

After the tool loop exits and the final response is committed:

```python
try:
    _engine.on_turn_end(messages)
except Exception as _engine_err:
    logger.debug("context engine on_turn_end failed: %s", _engine_err)
```

That's the entire framework change. Five lines plus a docstring.

## Backward compatibility

The default implementation is a no-op, so engines that don't override
`on_turn_end` (the built-in compressor, any third-party engine) are
unaffected. The runner's try/except means a buggy override degrades
gracefully — the agent keeps running, the issue is logged.

## Status

The patch is currently in-tree on the user's local hermes-agent
working tree. To upstream: open a PR with the two diffs above. Once
merged, autobio's runtime requirement on hermes-agent reduces to
"any version that includes the on_turn_end hook." Until merged, the
plugin requires a hermes-agent fork or an in-place patch.
