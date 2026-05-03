# Coordinate-System Cleanup Refactor

## Why

The projection-model refactor (see
`projection-model-refactor.md`) shipped V1 of the
archive-as-projection design. It worked — the Torment Nexus is gone,
hierarchies form correctly, render output validates clean — but the
last week of debugging surfaced a pattern that's worth naming.

Every bug we've fixed since the projection refactor has lived at the
boundary between three coordinate systems:

| System          | Identity                  | Used by                         |
|-----------------|---------------------------|---------------------------------|
| **Archive**     | content hash + position   | storage, mark_compressed         |
| **Filtered list** | sequential index         | picker (`_pick_chunk_bounds`)    |
| **Live view**   | render-time list index    | runner, validators               |

The picker walks a *filtered list* whose indices don't correspond to
archive positions; `mark_compressed` writes back by *hash* (which
stays correct); render walks *archive position*. Every fold-time bug
has been a translation error between these.

The bugs in chronological order:

1. **Markers archived as messages.** Render output went through
   `_archive_live` which couldn't reliably distinguish a render
   artifact from real input. Heuristic: content prefix + hash set.
2. **Order-blind tool-pair sanitization.** Render's tool-pair check
   pooled all tool_call_ids globally, missing OpenAI's "before next
   user/system" ordering rule.
3. **Tool-call shape mismatch.** Store flattens `{id, name, args}`,
   harness emits OpenAI nested form. Different content hashes for
   the same call → duplicate archive rows.
4. **Position cursor cached.** `_next_position` cached MAX(position)
   across compress() calls; new content reused old positions →
   duplicate positions in archive.
5. **Picker spans position gaps.** Filtered list compacts over
   already-covered messages. Picker selects "contiguous in filtered
   indices" but the corresponding archive positions skip a gap.
   Resulting L1 covers non-contiguous range. Render emits one marker
   per contiguous run.
6. **`protect_first_n` applied per-run.** Each fold left a
   3-message gap between itself and the previous fold. Consecutive
   L1s weren't position-adjacent → merged L2/L3 covered non-
   contiguous positions → duplicate markers (again).
7. **Dedup race after merge.** Speculative thread finishing AFTER a
   merge would create a duplicate-source_hash L1 because the dedup
   filter excluded merged summaries.

Each fix worked. But the rate at which new bugs appear at the same
boundary tells us the abstractions aren't carving the right joints.
The refactor below collapses most of these issues at the source.

---

## Goals

- **One coordinate system for the picker.** Walk archive position
  directly. No filtered list, no index translation.
- **Markers are structurally distinct from messages.** Identity by
  schema, not content heuristic.
- **One canonical tool-call shape.** No flatten/inflate dance.
- **Merge respects position contiguity.** Two L1s only merge if their
  combined coverage is position-adjacent.

Speculation stays. The under-lock dedup (already in) handles the
only race it can hit.

Non-goals for this refactor:

- Changing the as-of summarization invariants (already correct).
- Adding adaptive resolution per region (V2).
- Adding refold (V2).
- Changing CM ops or the dashboard (already correct).

---

## The four changes

### 1. Picker walks the archive directly

**Current state:**

```python
all_uncovered = _uncovered_messages(store)        # filtered list
uncovered = _latest_contiguous_run(all_uncovered) # 2nd-pass filter
msg_dicts = [_archive_row_to_msg(m) for m in uncovered]
chunk_start, chunk_end = _pick_chunk_bounds(
    msg_dicts, effective_protect_first_n, ...,
)
chunk_hashes = uncovered_hashes[chunk_start:chunk_end]
```

Three layers of translation. Bugs at each.

**Proposed:**

```python
chunk_hashes = _pick_archive_chunk(
    store, chunk_token_target, protect_last_tokens,
    weights_cfg=weights_cfg,
    is_eligible=is_eligible,
)
```

`_pick_archive_chunk` walks the archive in position order. State it
tracks during the walk:

- `current_run_hashes: List[str]` — the contiguous uncovered run
  we're accumulating
- `pending_chunk_tokens: float` — weighted token sum within the run
- protected ranges fetched once before the walk

Algorithm:

```
for row in archive_messages_ordered_by_position:
    if row.compressed_into is not None:
        # End of a run. Decide if current run is foldable; if so, emit
        # chunk and stop. Otherwise reset and continue.
        if current_run is eligible:
            return current_run sliced to chunk_token_target
        reset
        continue

    if row in pin_range and pin not eligible:
        if current_run is eligible:
            return current_run
        skip past pin
        continue

    accumulate row into current_run
    if pending_chunk_tokens >= chunk_token_target:
        return current_run

# end of archive
if current_run is eligible AND tokens >= chunk_token_target:
    return current_run
else:
    return empty (not enough content yet)
```

`protect_first_n` is applied ONCE: the picker's `start_position` is
`max(0, protect_first_n)`. After that boundary, every uncovered run
is fair game.

**What this fixes:** bugs #5, #6 disappear structurally. There's no
filtered list to compact over, no per-run protect_first_n, no
index-translation step.

**Migration impact:** moderate. Replaces `_pick_chunk_bounds` and
`_uncovered_messages`. Tests that call `_pick_chunk_bounds`
directly need updating (~3 tests). The legacy list-based
`compress_step` becomes a thin shim that materializes its messages
list into the archive then delegates.

---

### 2. Markers as a row_type column

**Current state:** markers are inserted into the live message list
during render via `_make_summary_marker`. They're recognized later
by content-prefix heuristic (`_MARKER_PREFIX = "[autobio:memory "`)
plus a hash-set check against active summaries. `_archive_live`,
`_reconcile_with_archive`, and the picker each have to know about
this heuristic.

**Proposed:**

Markers stay in render output (they're what the model sees) but
they're never archived. To recognize them on the way in, the render
output tags them via a private dict key:

```python
def _make_summary_marker(summary_id, level, tokens):
    return {
        "role": "user",
        "content": f"[autobio:memory {summary_id} · ...]",
        "_autobio_marker": True,    # explicit flag
    }
```

`_reconcile_with_archive` and `_archive_live` skip any message with
`msg.get("_autobio_marker") is True`. No prefix heuristic. No hash-set
fallback. The flag is dropped before sending to the LLM (or the LLM
simply ignores it; OpenAI clients pass through unknown keys).

If a future runner persists markers across process restarts (it
shouldn't — they're rendered each turn), the flag is lost on JSON
round-trip. Belt-and-braces: the prefix string check is a fallback,
logged at warning level if it fires.

**What this fixes:** bug #1 disappears structurally. The "is this
a marker?" question has a definitive answer in the dict itself.

**Migration impact:** small. Update `_make_summary_marker`,
`is_marker_message`, and the two archive paths to check the flag.
Existing tests don't assert on marker shape.

---

### 3. Canonical tool-call shape throughout

**Current state:** the store's `_normalize_tool_calls` flattens
OpenAI's `{id, type, function: {name, args}}` to `{id, name, args}`.
Render's `_msg_to_openai` re-inflates back. The flat form was
chosen at some point for "simplicity" but it's a footgun: any path
that bypasses the inflation step (the picker, validator, harness
input) sees a different shape and produces different content
hashes.

**Proposed:** store the OpenAI canonical form verbatim. JSON-encode
on insert, JSON-decode on read. Drop `_normalize_tool_calls`
entirely.

Schema migration: existing rows with flat-form tool_calls need
re-inflation. A one-shot migration in `_run_migrations`:

```python
for row in conn.execute("SELECT hash, tool_calls FROM messages WHERE tool_calls IS NOT NULL"):
    parsed = json.loads(row['tool_calls'])
    if parsed and isinstance(parsed[0], dict) and 'function' not in parsed[0]:
        # flat form — re-inflate to canonical
        canonical = [
            {"id": tc.get("id", ""), "type": "function",
             "function": {"name": tc.get("name", ""),
                          "arguments": tc.get("arguments", "")}}
            for tc in parsed
        ]
        conn.execute("UPDATE messages SET tool_calls = ? WHERE hash = ?",
                     (json.dumps(canonical), row['hash']))
```

**What this fixes:** bug #3 disappears structurally. Every code path
sees the same shape. Content hashes agree.

**Migration impact:** small for code, one-shot DB migration on first
open of an existing store. The migration is idempotent and read-modify-
write: safe to re-run.

---

### 4. Merge by position contiguity

**Current state:** `merge_if_ready` groups L1s by creation order
(consecutive IDs), `merge_threshold` at a time. It doesn't check
whether their coverage is position-adjacent. With pins, docs, or
rollback gaps creating uncovered "islands" between L1s, merges
produce L2s with non-contiguous coverage, which `render_view`
emits as multiple markers.

**Proposed:** at merge time, check that the L1s' position coverage
forms a contiguous range. If not, defer the merge until contiguity
exists (or never — sometimes pins are permanent).

Algorithm in `merge_if_ready`:

```python
unmerged = store.list_summaries(level, include_merged=False)
# Sort by position-of-first-source
unmerged.sort(key=lambda s: store.get_summary_turn_range(s["id"])[0] or 0)

# Group consecutive position-adjacent summaries
groups = []
current_group = []
for s in unmerged:
    coverage = positions_covered(store, s["id"])
    if not current_group:
        current_group = [s]
    elif positions_adjacent(current_group[-1]_coverage, coverage):
        current_group.append(s)
    else:
        groups.append(current_group)
        current_group = [s]
if current_group:
    groups.append(current_group)

# Merge any group that hits threshold
for group in groups:
    while len(group) >= threshold:
        batch = group[:threshold]
        new_id = _merge_batch(batch, level + 1, ...)
        group = group[threshold:]
```

`positions_adjacent(prev_coverage, next_coverage)` returns True iff
`prev_max_pos + 1 == next_min_pos` — no covered or uncovered messages
between them.

**What this fixes:** bugs #5 and #6 also have a downstream component
in merge. Even if the picker produces position-adjacent L1s,
*pre-existing* L1s from before the picker fix could still be
non-adjacent. This change is the structural backstop.

**Migration impact:** small. Existing non-contiguous merges from
prior runs stay merged (we don't unmerge). Future merges respect
contiguity.

---

## What about all the things we just patched?

Several recently-added helpers become unnecessary:

- `_latest_contiguous_run` — folded into the archive walk
- `effective_protect_first_n` — picker only protects archive
  position 0 onward, no per-run subtraction
- `_normalize_tool_calls` — gone
- `is_marker_message` content-prefix check — fallback only
- The `merged_into IS NULL` divergence between pre-LLM dedup and
  under-lock dedup — under-lock check stays as-is (just-fixed),
  which already handles the post-merge race correctly

The patches stay correct as long as they're in; the refactor just
removes the surface that needed them.

---

## Migration & rollout

**Feature-flag style isn't worth it.** The changes are small enough
in net code volume but they're all interrelated — gating them
individually creates a worse intermediate state.

Approach:

1. Branch off `main` (or current branch).
2. Implement changes 1–4 in one go, in this order: tool-call shape →
   marker flag → picker rewrite → merge contiguity. Each is local.
3. Run mock tests after each change.
4. Run the curated fuzz suite end-to-end after all four.
5. Re-run the self-replay (with redaction) and verify zero validator
   warnings.

Rollback: each change is independently revertable. If the picker
rewrite has issues, the other three can ship without it.

---

## Risks and open questions

1. **The picker rewrite may leak edge cases we haven't seen.**
   Specifically: pin-at-head, doc-spanning-multiple-runs, very long
   protected ranges. Mitigation: the existing fuzz scenarios cover
   all three. Run them as part of the validation gate.

2. **Schema migration for tool-call shape.** Re-running on an existing
   store rewrites those rows. If a session is currently active when
   the migration runs, in-flight `compressed_into` references stay
   valid (we update tool_calls only). Safe.

3. **Merge-contiguity might leave L1s permanently unmerged.** If a
   pin sits forever between L1-A and L1-B, they never merge.
   Acceptable: the L2 layer just has more L1s active; it's the
   correct semantics for "this region has a permanent split."
   Open question: should we surface this in the CM dashboard so
   the operator can see "merge blocked by pin at position X"?

4. **What about the `_archive_row_to_msg` / `_msg_to_openai`
   duplication?** They're already unified after my recent fix —
   one calls the other. Stays unified.

5. **Speculation thread shutdown timing.** Already handled (30s
   join). Not part of this refactor but worth re-verifying after.

6. **Backward compatibility with the existing replay6 store.**
   That store has flat-form tool_calls AND non-contiguous L2/L3
   coverage AND no markers (markers were never archived). Migration
   #3 fixes the tool_calls. The non-contiguous coverage stays
   harmless (rendering will emit multiple markers but no further
   harm). The viewer keeps working.

---

## Test plan

Existing tests that should pass unchanged after the refactor:

- All 27 mock tests (`test_autobio_plugin`, `test_autobio_thorough`)
- Curated fuzz suite (10 scenarios) — currently shows 0 errors with
  the patches in place
- Live API tests (`test_autobio_live`) — unchanged

New tests to add:

- `test_picker_walks_archive_skipping_covered` — synthetic
  archive with mid-archive coverage gap; picker should pick from
  the right run, not span the gap.
- `test_marker_flag_survives_render_round_trip` — render → harness
  appends a step → reconcile sees the marker via flag, skips it.
- `test_merge_contiguity_defers_non_adjacent` — three L1s with a
  pin gap between #2 and #3; merge_if_ready merges only #1+#2,
  leaves #3 unmerged.

Live-replay validation gate:

- Re-run the self-replay against the same Claude Code session JSONL
  (with redaction).
- Assertion: zero `duplicate_hash` warnings in the validator output.
- Eyeball check: a handful of summaries via the viewer.

---

## Estimated effort

- Change 1 (picker rewrite): ~3 hours including tests
- Change 2 (marker flag): ~30 min
- Change 3 (tool-call shape + migration): ~1 hour
- Change 4 (merge contiguity): ~1 hour
- Validation runs (mock + curated + replay): ~2 hours wall-clock
  (mostly LLM latency)
- Buffer for surprises: ~1.5 hours

Total: roughly half a day of focused work.

After this lands, the per-feature interaction surface is much
smaller. The bug rate I've been hitting at the picker/render
boundary should drop to roughly zero, because the boundary doesn't
exist anymore.

---

## Recommendation

Worth doing. The current patches work but the system has more
interaction-case surface than the spec implies it should. The
refactor is small in net code but topologically meaningful — most
of the bugs we chased this week become impossible by construction.

Speculation stays as designed. The dedup fix already in place is
the only race-handling needed.
