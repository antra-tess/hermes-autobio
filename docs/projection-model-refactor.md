# Projection-Model Refactor

## Why

The autobio engine drifts from the spec's first principle:

> The archive is authoritative; the live view is a projection.

In the current implementation the live view is **persistent mutated
state**. `fold_chunk` writes a marker into the in-memory message list
and the runner stores that mutated list as the new live state.
Markers persist forever, carry full L1 text inline (~1.4K tokens),
and accumulate across turns. They count toward the threshold, get
picked up as foldable chunks, and produce summaries-of-summaries
("Torment Nexus") indistinguishable from real conversation memory.

This refactor makes the live view what the spec says it is — a
**function of archive state**, recomputed each turn, never stored.

## Scope

V1 of this refactor:

- Archive becomes the only persistent state for compression.
- The live view is rendered fresh on every `compress()` call.
- Markers are a render-time artifact; they don't exist between calls.
- The picker walks **archive state** (uncovered messages), not the
  in-memory list.
- Merges, folds, and unfolds change archive state only. Live view
  catches up automatically on the next render.

Deferred to V2 (still useful, but not required to fix the Torment):

- Adaptive resolution per region (the `expanded_set` concept).
- Refold operation (inverse of unfold).
- Token-budget-aware automatic resolution choice (render at L2 if
  there's not room for L1s, etc.).

## Data model

**No new persistent tables required.** The data we need is already in
the schema:

- `messages` — every original message archived.
- `messages.compressed_into` — the L1 each message was folded into.
- `summaries.merged_into` — the L2/L3 each summary was rolled into.
- `pins` — pinned/document ranges.

Coverage of any message at any moment is derivable: walk
`compressed_into` → L1 → `merged_into` → L2 → `merged_into` → L3.
The "active" marker for a message is the highest non-NULL ancestor
whose own `merged_into` is NULL.

## New components

### `render_view(store, session_id, **opts) -> List[Dict]`

The single source of truth for what the model sees. Pure function:

```python
def render_view(store, session_id, *,
                protect_first_n=1,
                marker_format="brief"):
    """Return the live view as a list of OpenAI-format messages.

    Walks archived messages in chronological order. For each
    contiguous run of messages with the same active summary, emits
    one marker for that summary. Messages with no active summary
    (compressed_into IS NULL) are emitted as raw messages.

    Pure: no mutation, no LLM calls, no I/O outside the store.
    """
```

Render logic:

1. Query messages for the session in `position` order.
2. For each message, compute its `active_summary_id`:
   - If `compressed_into` is NULL → no active summary (raw).
   - Else walk: L1 → its `merged_into` (an L2) → its `merged_into`
     (an L3). Highest non-NULL whose own `merged_into` is NULL is
     the active marker.
3. Group consecutive messages by `active_summary_id`. For each group:
   - `active_summary_id` is None → emit each message verbatim.
   - Otherwise → emit one marker for that summary.
4. Sanitize tool pairs (existing helper).

### `picker_walk_uncovered(store, session_id) -> Iterator[Dict]`

The picker no longer walks a Python list of mixed messages and
markers. It iterates **archived messages with `compressed_into IS
NULL`**, in chronological order. There are no markers to skip
because there are no markers.

```python
def _pick_chunk_bounds_from_archive(store, session_id, *,
                                    chunk_token_target,
                                    protect_first_n,
                                    protect_last_tokens,
                                    protected_ranges,
                                    is_eligible,
                                    weights_cfg):
    """Same behavior as today's _pick_chunk_bounds, but operates on
    uncovered messages from the archive instead of a live list.
    Returns (chunk_message_hashes, doc_context_or_None).
    """
```

The boundary alignment, doc handling, and pin logic stay the same;
only the input source changes.

### Markers become tiny

A marker today is ~5K chars containing the full summary text. Under
the projection model markers are render-time only and exist purely so
the model knows "memory was here, you can call CM to expand it."
They should be cheap pointers:

```
"[memory L1-0042 · 800t · turns 92-101 · expand via cm_manage_context]"
```

~50 tokens. The full summary text is in the archive; CM can fetch it
via UNCOMPRESS / memory_read. The marker is just a structural anchor.

## Component-by-component changes

### `compression.fold_chunk`

- Was: mutate the live message list, return new list.
- Becomes: archive-only. Mark messages as `compressed_into = L1-N`
  and record the fact via `mark_compressed`. No list manipulation.
- Signature changes: returns the summary id, not a list.

```python
def fold_chunk(chunk_hashes, summary_id, store):
    # Validate summary exists & is active.
    # Call store.mark_compressed(chunk_hashes, summary_id).
    # That's it.
```

### `compression.compress_step`

- Was: pick chunk bounds in messages, summarize, fold the in-memory
  list, return list.
- Becomes:
  1. Pick chunk hashes from archive (`_pick_chunk_bounds_from_archive`).
  2. Summarize the chunk (existing `summarize_chunk` — operates on
     hashes, fetches actual content).
  3. Mark messages covered (`fold_chunk`).
  4. Optionally call `merge_if_ready`.
- Returns a `CompressResult` with summary_id + diagnostics. No
  message list in the return value.

### `engine.compress`

This is the runner-facing entry point. Existing signature returns a
message list, and the runner stores it. The new contract:

```python
def compress(self, messages, current_tokens=None):
    # 1. Reconcile: archive any messages from `messages` that aren't
    #    already in the archive. Skip stale markers (content starts
    #    with "[memory L"). New messages get current_turn.
    self._reconcile_with_archive(messages)

    # 2. Drive the compression step (archive-only).
    self._compress_step()

    # 3. Run merge_if_ready (also archive-only).
    merge_if_ready(self._store, self.cfg, self._model_info)

    # 4. Schedule speculation for the NEXT chunk.
    self.maybe_speculate()

    # 5. Render the live view from current archive state.
    return render_view(self._store, self._session_id,
                       protect_first_n=self.protect_first_n)
```

Reconcile logic:

```python
def _reconcile_with_archive(self, messages):
    for i, msg in enumerate(messages):
        if _is_marker(msg):
            continue   # stale render artifact — skip
        h = message_hash(msg)
        self._store.upsert_message(h, msg, turn=current_turn,
                                   position=archive_max_position + 1)
```

`_is_marker` recognizes by content prefix. Real user messages won't
start with `[memory L1-` etc., and even if they did, we'd just over-
archive (idempotent).

### `engine.on_turn_end`

Same pattern: archive new messages, schedule speculation, possibly
proactive CM. Doesn't need to return anything because the runner's
`messages` list is just a transient input.

### `merge_if_ready`

Already archive-only — writes the L2, marks L1s as merged. Just
works under the new model. The next `render_view` call will
naturally show the L2 marker covering what used to be three L1
markers.

### Picker

`_pick_chunk_bounds_from_archive` replaces `_pick_chunk_bounds`. It
walks archived messages with `compressed_into IS NULL`, in position
order. Same chunk-token accumulation logic, same protect_first/last
semantics, same pin/doc handling. Two consequences:

- **No markers to skip**, because the input has none.
- **No "head" computation either**: the picker's input is just
  uncovered messages, which by chronological-compression guarantee
  are at the trailing edge.

The summarizer prompt construction changes slightly:

- `[L3 summaries, oldest first]` — query store for active L3s.
- `[L2 summaries]` — query store for active L2s not under any L3.
- `[L1 summaries]` — query store for active L1s not under any L2.
- `[head]` — empty in the typical case (no uncovered messages
  earlier than the chunk by chronological invariant). For pinned
  regions or doc folding, fetch any covered-but-still-displayed
  content if/when we add adaptive resolution. **V1: head = empty.**
- `[marker, chunk, instruction]` — same as today.

### Speculative worker

Mostly unchanged. It already uses source_hash dedup on the archive.
Picks a prospective chunk via `_chunk_bounds_for` (which calls into
the picker). Switch its picker call to the archive-driven version.

### CM dashboard

The dashboard already queries the archive for layer counts and
previews. Pin info comes from the store. The dashboard is already a
projection — it just needs to render against the new "active
markers" model when describing the live view's structure.

### Tool result pruning

Operates on the live message list (the one the runner passes in).
This is fine: prune is a *display* concern that mutates the list
before it goes to the API. Under the new model, pruning runs after
`render_view` returns its output. The render is conceptually
"resolution choice"; pruning is "tool-result-specific resolution
choice." Same intent.

## Migration

This is a breaking refactor — `compress()`'s contract changes
slightly (still returns a list, but the list is now generated, not
mutated). The runner still works because it takes whatever
`compress()` returns as the new live list.

Steps:

1. Add `render_view` (new module, no callers yet).
2. Add `_pick_chunk_bounds_from_archive`, exercise via tests.
3. Add `_reconcile_with_archive` to the engine.
4. Add `_is_marker` helper to compression.
5. Switch `engine.compress` to the new flow.
6. Switch `fold_chunk` to archive-only.
7. Update existing tests (most tests pass messages and expect a list
   back; that's still valid).
8. Re-run mock + curated fuzz suites.
9. Then re-run replay against this session JSONL — should produce
   ~bound-by-archive number of L1s with no recursive Torment.

## What this fixes

- **Torment Nexus** — gone. Markers don't exist between turns, can't
  be picked.
- **Live list inflation** — gone. Tail size = tokens of uncovered
  messages + (small fixed cost per active marker).
- **L1 content duplicated in prompt** — gone. The L1's text appears
  exactly once in the prompt header, not also in the head.
- **Merge ↔ live view drift** — gone. Render queries current archive
  state every time.
- **Threshold firing on stale markers** — gone. Threshold measures
  uncovered + brief markers, not accumulated marker text.

## What this does NOT fix (deferred)

- **Adaptive resolution per region** (#6). Render always uses the
  highest active summary for each region. No "render this L2 as its
  3 L1s instead." That's the V2 expanded-set extension.
- **Refold** (no inverse of unfold yet — once unfolded, still in the
  archive, but no API to put the marker back).
- **Speculative timing** (#8). Still fires every `on_turn_end`.
- **Compression-generations confidence reporting in search results**
  (#9). Independent change.

## Risks / open questions

1. **Marker recognition by content prefix** is a heuristic. Real
   user content beginning with `[memory L` would be falsely treated
   as a stale marker and skipped on reconcile. Mitigation: include a
   distinctive sentinel (e.g., `\x01[memory L`) or store a separate
   flag. **Decision pending.**

2. **`tool_call_id` continuity.** When messages are folded and the
   marker takes their place in the rendered list, what happens to
   any in-flight tool-call IDs the assistant might still reference?
   The spec sanitizes tool pairs in `_sanitize_tool_pairs`. Render
   needs to do the same. Not new — same logic, just runs on the
   rendered output.

3. **Position numbers.** `messages.position` is the original
   archival index. After compression, the rendered list's indices
   are NOT positions — they're just list indices. Is anything
   relying on the live-list indices being "positions"? The picker
   used to; now it walks archive directly. Rest of the code should
   be index-agnostic. Verify.

4. **Backward compat for stored markers.** Existing sessions
   (replay6) have markers persisted in their last live state — but
   we don't actually persist the live state across process
   restarts. So this is a non-issue: every fresh process starts
   from the runner's initial messages and re-projects from there.

5. **CM ops that modify live view.** UNCOMPRESS in the spec is "show
   this L1 region as raw messages." Under V2's adaptive resolution
   that's "remove this summary from the rendered set." Under V1
   without adaptive resolution, UNCOMPRESS would have to mark the
   summary for refold-prevention, which we don't have. Either
   defer UNCOMPRESS to V2 or implement a minimal version (a session-
   scoped "do not auto-render this summary as a marker" set).

## Test plan

- Existing 27 mock tests should pass with minimal changes.
  `compress()` still returns a list; the list just looks slightly
  different (smaller markers, no accumulating bloat).
- New tests:
  - `test_render_view_groups_by_active_summary` — three L1s, one
    merged into L2, one not. Render produces an L2 marker covering
    the first L1 + the standalone L1 marker for the third.
  - `test_picker_skips_already_compressed` — fold messages 0-9 into
    L1-1. Add 10 new messages. Picker should pick from message 10
    onward, not re-encounter the L1-1 region.
  - `test_torment_nexus_does_not_recur` — synthetic scenario from
    the replay6 finding. After 50 forced compresses with no new
    content, summaries count should not grow without bound.

- Live test: re-run the self-replay against this same session JSONL.
  Assertion: number of L1s ≤ (uncovered messages / chunk_size). The
  current run produced 200+ L1s for ~900 messages (10x oversummary);
  fixed version should produce ~30-50.
