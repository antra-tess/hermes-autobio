# Required hermes-agent patches

The autobio plugin needs **two** framework hooks in hermes-agent:
``ContextEngine.on_turn_end(messages)`` and
``ContextEngine.on_llm_call(api_kwargs, response, *, error=None)``.
Without them, the engine has no per-turn visibility into the live
message list AND no per-call visibility into the prompt/response
the agent actually saw.

## Hook 1 — `on_turn_end(messages)`

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

## Hook 2 — `on_llm_call(api_kwargs, response, *, error=None)`

The autobio engine uses ``on_llm_call`` to record every main-agent
LLM call into the archive's ``llm_calls`` table. Without it, the
viewer's LLM-call records cover only plugin-internal calls
(summarizer, merge, CM); the main agent's actual prompts — which
is where memory-context rendering matters most — would have nowhere
to land for inspection.

### `agent/context_engine.py` — add the ABC method

```python
def on_llm_call(
    self,
    api_kwargs: Dict[str, Any],
    response: Any = None,
    *,
    error: Optional[Exception] = None,
) -> None:
    """Called once per LLM API call after the call returns.

    ``api_kwargs`` is the exact kwargs that went to the provider
    client (chat.completions.create / messages.create / bedrock
    converse). ``response`` is the provider's response object
    (None on failure; ``error`` carries the exception).

    Fires for every API call regardless of tool-loop iteration —
    one user turn produces N invocations of this hook for N
    tool-chained model calls. Default no-op; runner swallows
    hook errors.
    """
```

### `run_agent.py` — call it after each API response

In the main API-dispatch loop, immediately after the call returns
and before validation:

```python
if _use_streaming:
    response = self._interruptible_streaming_api_call(
        api_kwargs, on_first_delta=_stop_spinner
    )
else:
    response = self._interruptible_api_call(api_kwargs)

_engine = getattr(self, "context_compressor", None)
if _engine is not None and response is not None:
    try:
        _engine.on_llm_call(api_kwargs, response)
    except Exception as _engine_err:
        logger.debug(
            "context engine on_llm_call failed: %s",
            _engine_err,
        )

api_duration = time.time() - api_start_time
```

That's the second framework change. About ten lines.

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
