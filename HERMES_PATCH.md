# Required hermes-agent patches

The autobio plugin needs framework changes in hermes-agent across
three categories:

1. **Two new ContextEngine hooks** — ``on_turn_end(messages)`` and
   ``on_llm_call(api_kwargs, response, *, error=None)``.
2. **Engine-driven preflight compression** — the runner's preflight
   check must consult ``engine.should_compress_preflight(messages)``
   and bypass the length guard when the engine fires. This is what
   makes transfer-in / resume work: a session with seeded memory in
   the archive but only a few light recent messages on resume needs
   to fire compress() before the first API call so memory recall
   pairs land in conversation_history.
3. **System-prompt tuning knobs** (optional but strongly recommended
   for autobio deployments) — ``agent.skip_default_identity`` plus
   per-tool guidance overrides ``agent.{memory,session_search,skills}_guidance``.
   Without these, the autobio agent's prompt carries Hermes's
   identity and a memory-tool guidance block that talks about a
   subsystem autobio replaces.

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

## Change 3 — Engine-driven preflight compression

The runner's preflight-compress check at session-resume time is
gated on a length guard:

```python
if (
    self.compression_enabled
    and len(messages) > self.context_compressor.protect_first_n
                        + self.context_compressor.protect_last_n + 1
):
    ...token threshold check...
```

For an autobio transfer-in (session resumed with seeded memory in
archive but only 1-2 short messages in conversation_history), the
length guard skips the block entirely and the engine never gets a
chance to render its memory recall pairs into conversation_history
before the first API call.

### `run_agent.py` — consult the engine alongside the length guard

```python
_has_min_messages = (
    len(messages) > self.context_compressor.protect_first_n
                    + self.context_compressor.protect_last_n + 1
)
_engine_wants_preflight = False
if self.compression_enabled:
    try:
        _engine_wants_preflight = (
            self.context_compressor.should_compress_preflight(messages)
        )
    except Exception as _pf_err:
        logger.debug(
            "context engine preflight hook failed: %s", _pf_err,
        )
if (
    self.compression_enabled
    and (_has_min_messages or _engine_wants_preflight)
):
    _preflight_tokens = estimate_request_tokens_rough(...)
    _threshold_hit = (
        _has_min_messages
        and _preflight_tokens >= self.context_compressor.threshold_tokens
    )
    if _threshold_hit or _engine_wants_preflight:
        ...compress...
```

Preserves existing behavior for all token-driven cases; only adds
the engine-fire path. Built-in `ContextCompressor`'s default
`should_compress_preflight` returns False so it's never affected.

## Change 4 — System-prompt tuning knobs (optional)

Three small additions to `_build_system_prompt` and the
``agent`` config section:

### Default-identity off-switch

Skip the SOUL.md / DEFAULT_AGENT_IDENTITY block (the
"You are Hermes Agent…" identity layer) so the agent's identity
can come from memory or a transferred-in agent's accumulated
context.

```yaml
# config.yaml
agent:
  skip_default_identity: true
```

### Per-tool guidance overrides

The default tool-guidance blocks (`MEMORY_GUIDANCE`,
`SESSION_SEARCH_GUIDANCE`, `SKILLS_GUIDANCE`) are appended to
the system prompt only when the corresponding tool is loaded.
For autobio deployments — where the built-in `memory` tool is
replaced by autobio's recall-pair mechanism — the
`MEMORY_GUIDANCE` block becomes inaccurate. Add three config
keys, each accepting:

- non-empty string: replaces the default text verbatim.
- empty string: suppresses the block entirely (still gated on
  the tool being loaded — but injects nothing).
- unset / None: keeps the default text.

```yaml
# config.yaml
agent:
  memory_guidance: ""           # disable — autobio handles memory
  session_search_guidance: ""   # disable — autobio's archive search supersedes
  skills_guidance: ""           # tune to taste
```

These are small additions to `_build_system_prompt`'s tool-
guidance loop. The default-identity switch additionally requires
passing `skip_soul=True` to `build_context_files_prompt` when the
identity is suppressed, so SOUL.md doesn't sneak back in via the
context-files path.

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
