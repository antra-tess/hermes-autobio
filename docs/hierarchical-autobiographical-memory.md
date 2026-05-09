# Hierarchical Autobiographical Memory System

## Design Specification

A memory system for Hermes Agent that treats conversation history as
autobiographical experience. Raw messages are never discarded — they
are progressively folded into layered summaries that preserve the
subjective arc of the conversation. A dedicated Context Manager (CM)
agent mediates between the main instance and the memory layers.

This document is as much a record of **why** as it is of **what**.
The design is full of choices that look arbitrary until you see the
gotcha they prevent. The `Design decisions — why not otherwise`
section near the end captures the pitfalls so future readers don't
rediscover them.

---

## Terminology

Two distinct concepts that early drafts of this design (and the v1
implementation) confused under the single word "marker." Resolved:

- **Marker** — the in-band priming text inserted into the
  *summarizer prompt* between head and chunk. The string is "System:
  You will soon form a new memory, get ready..." (`_MARKER` in
  `compression.py`). It tells *the summarizer* "the next slice is
  what you're forming a memory of." Visible only during memory
  writing; never appears in the main agent's live view.

- **Recall pair** — the live-view representation of a folded region.
  A two-turn dialogue placed where the original messages used to be:
  `[user "[CM] Recall memory L1-X (gen N)."]` followed by
  `[assistant "<first-person body>"]`. CM (in user role) asks the
  agent to recall the memory; the agent (in assistant role)
  produces the recall — the memory's content, exactly as the
  summarizer wrote it. This is how memories arrive in any LLM call
  the agent makes, not just summarizer prompts.

The two concepts have nothing in common except the unfortunate
shared name in earlier text. When this doc says "marker" without
qualification it means the summarizer-prompt priming text.

---

## Core principles

Four invariants everything else follows from. Read these first.

### 1. The archive is authoritative; the live view is a projection.

The SQLite archive holds every message, every tool result, every
summary ever produced. It only grows. The conversation list fed to the
model is a **view** of that archive at a chosen resolution, bounded by
the context window. Nothing is ever irreversibly removed — compression
means *render at lower resolution in the live view*, not *destroy*.

### 2. Compression is always chronological and never destructive.

Oldest uncompressed content folds first. We never retroactively
compress a middle range while leaving an earlier range uncompressed.
This keeps the as-of invariant (principle 4) trivially satisfied —
every committed summary was formed with the correct historical
context.

### 3. Pins are display locks, not write locks.

A pin says "keep this range at raw resolution in the live view." It
does **not** block summarization. We can (and often should)
speculatively summarize pinned content in the background so the L1 is
ready the moment the pin releases under pressure. Pinning defers the
swap; it does not defer the archive.

### 4. Summarization preserves the as-of moment.

When we write a memory for chunk C, the summarizer's context must
match what the instance saw *when C was the live tail*. Nothing from
after C is visible. Including later content leaks future information
into the KV state and poisons the memory with retrospective framing
("I read the log, which *turned out* to be relevant to the bug I
found next"). The memory must read as formed in the moment.

---

## Architecture Overview

Two Hermes plugin interfaces are used:

- **ContextEngine** (`plugins/context_engine/autobio/`) — compression
  lifecycle, speculative summarization, CM agent loop, per-turn hook.
- **MemoryProvider** (`plugins/memory/autobio/`) — search tools,
  archive indexing, turn-counter bookkeeping.

They share a single SQLite database accessed through a Store
singleton. The two plugin directories each contain a thin
`register(ctx)` shim; the real code lives in the sibling package
`plugins/autobio/` so both halves can import from it.

Four small framework changes upstream in hermes-agent:

- `ContextEngine.on_turn_end(messages)` — per-turn message-list
  visibility for speculative scheduling + proactive CM.
- `ContextEngine.on_llm_call(api_kwargs, response, *, error)` —
  per-call hook for recording the main agent's exact prompt/response.
- Engine-driven preflight via `should_compress_preflight(messages)`,
  consulted by the runner alongside the length guard. Lets
  transfer-in surface recall pairs on resume even when the live
  message list is short.
- System-prompt tuning knobs (`agent.skip_default_identity` plus
  guidance overrides for memory / session_search / skills) so the
  agent's prompt assembly can defer to autobio's recall mechanism
  instead of fighting the default Hermes identity layer.

All four are backward-compatible (no-op defaults). See the Hermes
Interface Mapping section, and `HERMES_PATCH.md` for full diffs.

---

## Memory Layers

```
L0  Raw messages          The uncompressed tail (~100k tokens)
L1  Recent memories       ~10k token chunks folded to ~2-3k
L2  Consolidated          6 L1s merged into 1 L2
L3  Long-term             6 L2s merged into 1 L3
```

Layers are not fixed strata — they are **adaptive resolution**. The
memory landscape at any moment might look like:

```
...L3...L3...L2...L1...L1...L2...L3...L2...L1...L0 (tail)
```

Old memories can be unfolded on demand, creating valleys of high
fidelity deep in the past. Recent trivia can be folded early. The
topography is shaped by what matters — the main instance's attention
and CM's judgment together sculpt which regions get full resolution
and which stay summarized. The archive always has every layer
available; the live view picks one resolution per region.

The constraint is token budget: unfolding a valley means folding
something else or accepting a shorter tail. This is exactly what CM
negotiation handles — trading resolution in one region for resolution
in another, always within the context limit.

### Compression Ratios

| Transition | Input            | Output          | Ratio |
|-----------|------------------|-----------------|-------|
| L0 → L1   | ~10k token chunk | ~2-3k summary   | ~4:1  |
| L1 → L2   | 6 L1s (~15k)     | 1 L2 (~3k)      | ~5:1  |
| L2 → L3   | 6 L2s (~18k)     | 1 L3 (~4k)      | ~4:1  |

### Storage

```
~/.hermes/memory/autobio/
├── store.db           # SQLite: messages, summaries, pins, FTS5
└── summaries/
    ├── L1/
    │   ├── L1-0001.md
    │   └── ...
    ├── L2/
    └── L3/
```

Summary markdown files live alongside the DB so raw-text `grep` works
on them without going through SQLite. The DB has FTS5 virtual tables
indexing both messages and summaries for BM25 search.

---

## The two operations

The core of the system is a split that took a round of design to
arrive at. Early drafts fused summarization and display mutation into
one `compress()` call. That fusion makes pins impossible to reason
about and blocks background work. Splitting the primitives:

### summarize_chunk — archive write

Given a chunk of messages, produces an L1 and writes it to the
archive. **Does not touch the live view.**

- Safe to run eagerly, including for pinned content
- Safe to run in the background
- Idempotent via `source_hash` lookup: if an L1 already exists for
  this chunk's source_hash, returns that L1's ID without calling the
  LLM. The archive IS the cache.

### fold_chunk — display mutation

Given a chunk and its already-committed summary ID, replaces the
chunk in the live message list with a **recall pair** (CM-voice user
header + agent-voice assistant body). No LLM call.

- This is the decision gated by pins, documents, and token pressure
- Callers should run summarize_chunk first (or verify via
  `find_summary_by_source_hash`); fold requires the summary to exist
- Sanitizes orphan tool_call/result pairs post-swap

### unfold — display restoration

Inverse of fold. Given a summary ID, locates its recall pair in the
live list and expands back to source messages from the archive. For
L1s the source is raw messages; for L2/L3 it's the constituent L1s'
recall pairs (the "fill lower orbitals first" rule means L2/L3 are
themselves rendered as their L1 children's recall pairs).

- Works at any time regardless of fold state
- Token budget may require folding something else first; CM mediates

### compress_step — orchestrator

What the engine's `compress()` actually calls: pick chunk bounds,
summarize, fold. Preserves the previous atomic signature for callers
that don't care about the split.

---

## The KV-preserving prompt

The summarization LLM call is constructed to match what the instance
saw at the as-of moment. **Strict chronological order, nothing after
the chunk**:

```
[L3 summaries, oldest first]
[L2 summaries]
[L1 summaries]
[head — messages before the chunk]
[marker: "System: You will soon form a new memory, get ready"]
[chunk — the slice being compressed, NEWEST in this context]
[instruction: "Write the memory of events since the most recent
 memory system notification"]
```

The chunk is the newest thing the summarizer sees. There is
intentionally no `tail_after_chunk` — that would be future
information.

The marker is an in-band signal, not a system prompt. It primes
attention without disrupting conversational KV state. Subsequent
compressions refer back to the same marker wording so the model
treats memory formation as a recurring narrated event rather than a
fresh instruction each time.

### Why no tail-after-chunk?

At the as-of moment when the chunk was the live tail, messages after
it didn't exist. Including them in the summarizer's context is
telling the past the future. The summary inherits retrospective
framing: it might say "I began debugging the auth module, which
*turned out to be* caused by an issuer-suffix mismatch" — folding in
knowledge the instance couldn't have had. The memory now reads like a
post-mortem, not a lived experience. That's a KV disturbance the
whole design exists to avoid.

Compression is strictly chronological (principle 2), so all existing
summaries in the store predate the chunk by construction. No
timestamp filtering needed — the store state at summarization time IS
the correct as-of context.

### Chunk Boundary Selection

Chunks are anchored to stable message IDs (content hashes), not
positional indices, so rollbacks don't cascade invalidation.

Picker walks forward from `protect_first_n` accumulating ~`chunk_tokens`
of content. Both ends align to avoid splitting an assistant/tool_result
group. Leaves at least `protect_last_tokens` worth of messages in the
tail; if not possible, returns empty range and compression skips.

When the candidate chunk overlaps a protected range (pin or
document), behavior depends on kind — see Pins and documents.

---

## Speculative summarization

A background worker runs summarization ahead of time so when
compression actually fires, the L1 is already in the archive.

### How reuse works

1. When tail approaches threshold (triggered via `on_turn_end`), the
   worker picks the prospective chunk and its source_hash.
2. If an L1 already exists for that source_hash (found via
   `find_summary_by_source_hash`), nothing to do — archive is the cache.
3. Otherwise, snapshot the current state, run `summarize_chunk` in a
   background thread. The result commits to the archive.
4. When the real `compress_step` fires, its `summarize_chunk` call
   finds the existing L1 and skips the LLM. Zero-latency compression.

### Rollback behavior

Rollback resilience is a natural consequence of how source_hash works:

- `/undo` that removes messages **after** the chunk → chunk is
  unchanged → source_hash unchanged → existing L1 is reused. The
  summary is a valid memory of that chunk regardless of what came
  after.
- `/undo` that reaches **into** the chunk → new source_hash. The old
  L1 stays in archive (harmless — a memory of a chunk that no longer
  appears in the live view). A fresh L1 is written for the new chunk.

This was the point of switching from a combined-hash-of-everything
cache to source_hash-only dedup: the memory stays valid even as
context downstream shifts. The archive is immutable; the live view is
fluid.

### Sequentiality

Only one speculative summarization in flight at a time. L1 #N+1
depends on whether L1 #N was accepted, so parallelism here is unsafe.

---

## Pins and documents

Protected ranges are a unified concept. A **pin** is a range of
messages the live view should keep at L0 (raw). A **document** is a
pin with extra metadata describing *how* to fold it when pressure
eventually forces the issue.

Both live in the `pins` table with columns:

```sql
pins (id, session_id, start_hash, end_hash, reason, created_at,
      kind, meta)

kind IN ('pin', 'doc_chronological', 'doc_monolithic')
meta  TEXT  -- JSON
```

### What "pin" means

A pin on a range says: *in the live view, keep this range rendered at
raw L0*. It does not block summarization — speculative worker may
still produce an L1 for the range, tucked safely in the archive.
Pinning is a display decision, not a write decision.

This is the subtle part. Earlier drafts treated pins as "never
compress" which meant pinned content stayed raw until it dominated
the tail and crowded out everything else. Under the new model, the
archive grows as usual; only the display is locked. When pressure
eventually forces the pin to release (or relax from L0 to L1), the
summary is already waiting.

#### Range-pin vs. summary-pin

Two distinct things in the schema both got called "pin." Disambiguate:

- **Range-pin** lives in the `pins` table (`kind='pin'`). Covers a
  span of message hashes. Read by compression's fold-protection logic
  to keep raw messages from being folded. Exercised by fuzz scenarios
  via direct `add_pin(... kind='pin')` calls. **Live and used.**
  Currently no CM op exposes range-pin creation directly — only
  document pins (`kind='doc_*'`) get a CM-facing op via
  `MARK_DOCUMENT`. Range-pin support is a hook for future CM ops or
  programmatic protection of specific spans.

- **Summary-pin** is the `summaries.pinned` boolean column. Was
  intended to protect a specific L1/L2 from being merged into a
  parent. **Vestigial under v1**: the balance rule means render
  always emits L1 recall pairs regardless of merge state, so the
  flag has no observable effect. The CM ops that read/write it
  (`PIN <id>` / `UNPIN <id>`) are not exposed in v1 (see CM
  Operations section). The column and op handlers are kept in the
  codebase but return an explanatory error so a CM that hallucinates
  the op from training data gets a clear signal. Re-introduced in
  v2 when adaptive resolution lets pinning actually keep a region
  at higher fidelity than its merge state would dictate.

When this doc says "pin" without qualification it means **range-pin**.

### Documents

A document is a chronological narrative that cannot reasonably be
removed from the live view without distorting the instance's
continuity. Think: a long log file the instance just read; a
multi-hour transcript; a report. These often exceed the context
budget entirely (>200k tokens).

Documents cannot stay raw forever. They'll eventually need to fold —
but the fold must happen *gracefully* and *chronologically*. For
chronological docs:

- **Oldest part folds first.** The doc's "top edge" advances as
  chunks summarize away. Later parts of the doc stay raw until it's
  their turn.
- **The as-of principle applies per chunk.** When doc chunk N is
  being summarized, the summarizer sees pre-doc conversation, the L1s
  of doc chunks 1..N-1 (from earlier folds), marker, chunk N,
  instruction. It has not yet read chunk N+1 and should not see it.
- **Boundary snapping.** A 10k-token slice inside a log should snap
  to a paragraph break or timestamp rather than cut mid-entry.

For monolithic docs (where chronological slicing would lose meaning —
a non-chronological reference document, for example), the doc is
treated atomically: fold the entire thing or leave it raw. There's no
clean middle.

### Eligibility policy

When is a doc eligible to fold? Default is **size pressure**:

- Non-doc chunks get folded first
- Only when no non-doc capacity remains AND tail usage is above some
  threshold (default 85%) do we start eating into documents
- Chronological docs fold from the top, one chunk at a time

Alternative policies: `age` (fold after N turns since creation),
`manual` (only CM explicitly requests fold via an op). Selected per
instance via `autobio.documents.eligibility`.

### Chunk picker integration

The chunk picker consults active pins. When walking forward to find
the next chunk:

- Candidate chunk inside a `pin` range → advance past its end_hash,
  try the next slice
- Inside an ineligible doc → advance past its end_hash
- Inside an eligible `doc_chronological` → clamp the chunk from the
  doc's current start_hash forward to ≤ chunk_tokens, snapping to a
  natural boundary
- Inside an eligible `doc_monolithic` → use the entire range as one
  chunk

### Doc self-retirement

When a chronological doc's top chunk folds:

1. `fold_chunk` runs as usual
2. The pin's `start_hash` advances to the next doc message
3. If `start_hash == end_hash`, the whole doc has folded — the pin row
   is deleted. The L1s remain in the archive and can still be unfolded.

### Why not a separate `documents` table?

Earlier drafts imagined a `documents` table, a `doc_id` on summaries,
and bespoke document queries. We retracted that. A document is
*exactly* a pin with compression metadata. Style is an attribute;
auto-detection is a creation-time concern; top-edge tracking is
just `start_hash` advancement. None of these need a new table. Making
documents a kind of pin keeps the lookup logic shared and the concept
unified.

---

## Tool Result Management

Without active management, tool results dominate the context window.
A few hours of active tool use can easily produce 500k+ tokens of
tool output — dwarfing the actual conversation. Tool result pruning
is not optional; it is the primary mechanism for keeping the tail
usable.

### Archive-First Principle

Every tool result and every tool call argument is stored in full in
the SQLite archive **before** any pruning touches the live list. The
archive is the permanent record; the live list is a budget-managed
view.

On every tool result:

1. Store full result content in SQLite, keyed by `tool_call_id` and
   content hash
2. Store full tool call arguments alongside
3. Index for grep (raw text) and FTS5 keyword search
4. Then apply pruning to the live message list

### Pruning Pipeline

Three passes, archive-first and non-destructive:

**Pass 1 — Deduplication**. Identical tool results (same file read
multiple times, same command re-run) are collapsed. Older copies
replaced with `[Duplicate — same content as a more recent call]`.
The newest copy survives in full.

**Pass 2 — Result summarization**. Tool results outside the per-tool
retention window and exceeding `summary_min_chars` are replaced with
informative 1-liners:

```
[terminal] ran `npm test` -> exit 0, 47 lines output
[read_file] config.py from line 1 (3,400 chars)
[search_files] 'rate_limit' in src/ -> 8 matches
```

Generated from tool name + arguments + result metadata — no LLM call.

**Pass 3 — Argument truncation**. Tool call arguments in assistant
messages (e.g., `write_file` with 50k of content) are truncated
beyond `argument_truncate_chars`. The most recent call of each tool
is exempt — the assistant may still be reasoning about what it just
sent.

### Retention Windows

Each tool type has a configurable retention window in tokens.
Retention is computed **per-tool-type**: walking backward, we
accumulate content tokens separately for each tool. Terminal can hold
50k of recent output even if file reads and searches have pushed
everything else back.

```yaml
tool_pruning:
  default:
    full_retention_tokens: 30000
    summary_min_chars: 200
    argument_truncate_chars: 500
  per_tool:
    terminal:
      full_retention_tokens: 50000
    read_file:
      full_retention_tokens: 15000
    memory_search:
      full_retention_tokens: 10000
    write_file:
      argument_truncate_chars: 200
  dedup: true
```

### Uncompression from archive

When the main instance needs a pruned tool result back, CM looks it
up by `tool_call_id` in the archive and returns the full result. The
main instance never needs to know whether a result was pruned; CM
handles recovery transparently.

### Interaction with hierarchical compression

When messages enter L1 summarization, the summarizer sees the
**pruned** version of tool results. This is intentional — the
summarizer doesn't need 50k of terminal output to write "ran the test
suite, 14 tests failed, mostly in auth." The full output remains in
the archive for grep/search if exact details are ever needed.

---

## Context Manager

CM is a dedicated Sonnet-class instance (configurable) that manages
the memory layers on behalf of the main instance. It is both a tool
the main instance can invoke and a proactive agent that wakes up
every N turns.

### CM as Tool

The main instance can call CM at any time:

```
cm_manage_context(request: str)
```

Examples:

- "I need the full auth discussion from earlier today"
- "Compress aggressively, starting a new topic"
- "What do we have on rate limiting?"
- "Pin the last 20 turns, I'll need them"
- "Set check-in interval to 100 turns"

CM receives the dashboard (current layer counts + summary previews +
pins + session info + recent topic signals), translates the natural
language request into formal operations, executes them, and returns
a natural-language response to the main instance.

### CM Proactive Check-ins

CM wakes up every N turns (default: 50). On wake:

1. Read the dashboard
2. Decide: INJECT or DEFER
   - If recent turns are dense tool-call chains → defer
   - If the conversation is at a natural pause → inject
   - Defer returns `DEFER N <reason>` and reschedules

3. If injecting, formulate a contextual check-in informed by dashboard
   stats — not generic, but specific to current state

Proactive check-ins stay in the conversation history as regular
assistant messages. They are genuine metacognitive activity and
become part of the autobiographical record.

### CM Dashboard

Every CM call receives a YAML dashboard:

```yaml
tail:
  tokens: 94200
  message_count: 289
context:
  usage_percent: 78
  limit: 200000
  threshold: 150000
layers:
  l1:
    count: 14
    total_tokens: 28000
    newest: L1-0014
    recent:  # previews of the last N per layer
      - id: L1-0014
        first_turn: 232
        last_turn: 254
        preview: "We decided to use Redis for rate limiting..."
  l2: { count: 3, total_tokens: 9000, recent: [...] }
  l3: { count: 1, total_tokens: 3000, recent: [...] }
pins: [...]
documents: [...]     # active pins with kind=doc_*
activity:
  current_turn: 289
  checkin_interval: 50
  turns_since_checkin: 34
archive:
  messages_stored: 1842
```

**Why previews matter**: early drafts of the dashboard showed layer
*counts* but not *content*. CM ended up writing confident replies
like "I don't have information about rate limiting..." and then
immediately running a SEARCH op that surfaced the info right below.
The narrative contradicted the ops because CM was reasoning from
counts alone. Previews let CM see what's actually there and answer
directly when the dashboard already has the answer.

### CM Operations

| Op                       | Effect                                                            |
|--------------------------|-------------------------------------------------------------------|
| `SEARCH "q"`             | FTS5 hybrid search over summaries + messages                      |
| `GREP "pat"`              | Raw text grep across summary markdown on disk                    |
| `UNCOMPRESS <id>`        | Render `<id>`'s source as raw messages in the live view, AND inline the expansion in the op result so the main instance sees it now. The unfolded state persists across renders (stored in the `unfolded_summaries` KV) until explicitly refolded |
| `REFOLD <id>`            | Inverse of UNCOMPRESS: revert the live view to showing `<id>` as a recall pair instead of raw messages |
| `MARK_DOCUMENT <s> <e> [style] [title]` | Register a protected range as a document (style: `chronological` or `monolithic`) |
| `LIST_DOCUMENTS`         | Enumerate active documents with progress (chunks folded vs. remaining) |
| `UNMARK_DOCUMENT <pin>`  | Retire a document pin; content stays archived                     |
| `MERGE <level>`          | Force a merge pass at the given level (usually automatic)         |
| `SET_INTERVAL <N>`       | Change CM's proactive check-in frequency                          |
| `DEFER <N> "reason"`     | Skip this check-in, reschedule `N` turns ahead                    |
| `RESUME_COMPRESSION`     | Clear a halt flag set when the summarizer returned empty after retries (use only after the underlying cause has been addressed) |

Notably **absent**: a way to compress arbitrary middle ranges. CM
does not direct the compression schedule — compression is always
chronological. CM directs **display resolution**: which layer to
render a region at, which ranges to unfold, which ranges to protect
as documents.

**Deferred**: summary-level `PIN <id>` / `UNPIN <id>`. The
``summaries.pinned`` column exists, but under the v1 balance rule
``render_view`` walks ``messages.compressed_into`` directly and emits
a recall pair for every L1 (active or merged) regardless of pin
state — so summary-level pinning has no observable effect on the
agent's prompt. Re-introduced in v2 when adaptive resolution makes
merge-state actually control render fidelity. For range-level
protection of raw messages, use `MARK_DOCUMENT` instead.

### Op ordering rule

CM's natural-language reply is shown to the main instance **before**
the operations run. This means CM must not make confident claims
about absence ("I don't have information on X") without having
already seen it in the dashboard. If CM intends to SEARCH, it should
frame the reply as "let me check" and state the findings from the
op results.

This is enforced by explicit instruction in CM's system prompt. The
correct pattern:

```
<reply text: "let me check the memory for rate limiting">

OPS:
SEARCH "rate limiting"
```

The main instance then sees CM's reply followed by the op results —
a coherent narrative.

### Multi-turn negotiation

When the main instance requests an uncompress that exceeds the token
budget, CM can negotiate:

```
Main: "I need the full Docker debugging session back"
CM:   "That's 45k tokens. Current tail is 94k, limit is 200k with
       layer overhead. I can give you the full session if I fold
       30k of your oldest tail first. Or just the resolution part
       (12k). Which do you prefer?"
Main: "Fold the old tail, give me the full session"
CM:   [folds, then unfolds the Docker L1]
      "Done. Docker session is now in your tail (turns 88-142)."
```

This exchange stays in conversation history as regular messages.

---

## Search System

### FTS5 keyword + raw grep

The search layer is intentionally minimal in v1:

- **SQLite with FTS5** virtual tables over messages and summaries —
  BM25 scoring over keyword terms
- **Raw grep** over summary markdown files on disk — for exact
  matches that keyword search misses (function names, error codes,
  config values, URLs)

No embeddings. See Design decisions below for why.

### Tools exposed

- `memory_search(query, max_results)` — BM25 over all layers
- `memory_grep(pattern, max_results)` — regex over summary files
- `memory_read(id)` — full content of a summary or archived message

### Provenance in search results

Results include the layer and compression generation:

```json
{
  "source": "L2-0006",
  "level": "L2",
  "compression_generations": 2,
  "first_turn": 44,
  "last_turn": 67,
  "snippet": "We chose Redis for rate limiting...",
  "score": 0.84
}
```

CM surfaces `compression_generations` to the main instance as a
confidence signal ("L2, gen 2 — some nuance may be lost").

---

## Proactive Features

### Surfacing (deferred to v2)

Detect when the main instance is referencing something in compressed
memory and offer to unfold it. Cross-reference topic mentions in
recent turns against L1/L2/L3 content; offer unfold when a topic
appears 3+ times in the last 20 turns.

### Forgetting confirmation (deferred to v2)

Before a speculative compression executes, optionally interrupt:
"The React performance discussion from 1h ago is about to become L1.
Want to keep anything in full?" Off by default — disruptive for most.

### Contradiction detection (deferred to v2)

When the main instance makes a decision, CM cross-references recent
statements against L1/L2/L3 entries and flags high-similarity matches
with opposing sentiment. Needs embeddings or an LLM-in-the-loop
pass, so deferred alongside embeddings.

### Confidence decay and resolution control

CM reports compression generation count when surfacing memories:

- From L1 → "first-hand memory, high fidelity"
- From L2 → "consolidated from 6 memories, some nuance may be lost"
- From L3 → "twice-consolidated, broad strokes only"

But compression is never permanent loss. When fidelity matters, CM
can unfold in place — expanding an L3 back to its L2 constituents,
or further back to L1s, or further to raw messages. This creates a
valley of high resolution surrounded by compressed context:

```
Main: "I'm not confident in my memory of that auth decision,
       the summary feels too thin"
CM:   "That's L3-0002, compressed twice. I can expand it back to
       the three L2s it came from — about 9k tokens. I'll fold
       the Docker debugging section to make room."
Main: "Do it"
CM:   [folds turns 180-200 into L1, unfolds L3-0002 into
       L2-0004, L2-0005, L2-0006 in place]
```

The main instance now has detailed memory of the auth decision
surrounded by compressed context on both sides. When it's done with
that region, CM can refold it — or leave it expanded if the instance
keeps referencing it.

### Catastrophic recovery (deferred to v2)

There are no "new sessions" in this paradigm. The agent is long-lived.
A context reset (crash, forced restart) is catastrophic — the
uncompressed tail is lost.

On recovery, CM reconstructs what it can from the archive: reload
recent L1s, inject them as context, notify the main instance of the
gap. The main instance operates in degraded mode knowing continuity
broke.

This is damage control, not a feature. The system should be designed
to avoid resets entirely.

---

## Hermes Interface Mapping

### ContextEngine implementation

```python
class AutobioContextEngine(ContextEngine):
    name = "autobio"

    def should_compress(self, prompt_tokens):
        # Fire when tail exceeds configured threshold

    def should_compress_preflight(self, messages):
        # Engine-driven preflight: always fire when memory exists in
        # the archive, regardless of message count. Lets transfer-in
        # surface recall pairs on the first turn even when the live
        # list is short.

    def compress(self, messages, current_tokens=None, **kwargs):
        # Archive-first, then prune, then compress_step orchestrator.
        # **kwargs absorbs runner-passed extras (e.g., focus_topic)
        # so ContextEngine API additions don't break us.

    def on_turn_end(self, messages):
        # Hermes ABC hook (new in autobio). Per-turn visibility into
        # the live list. Used for speculative scheduling + proactive
        # CM check-in.

    def on_llm_call(self, api_kwargs, response, *, error=None):
        # Hermes ABC hook (new in autobio). Records the main agent's
        # exact prompt + response into the ``llm_calls`` table with
        # ``purpose="main"``. Handles OpenAI, Anthropic-native, and
        # streaming-reconstructed response shapes. Without this, the
        # viewer would only show plugin-internal LLM calls and the
        # main agent's prompts — the actual memory-context rendering
        # surface — would have nowhere to land for inspection.

    def get_tool_schemas(self):
        # Expose cm_manage_context tool

    def handle_tool_call(self, name, args, **kwargs):
        # Route to CM via cm_respond()

    def on_session_start(self, session_id, **kwargs):
        # Open store, init speculative worker, register in active
        # engines dict so the provider can reach us

    def on_session_end(self, session_id, messages):
        # Final archive pass, shutdown worker, close store
```

### MemoryProvider implementation

```python
class AutobioMemoryProvider(MemoryProvider):
    name = "autobio"

    def initialize(self, session_id, **kwargs):
        # Open store (shared singleton with engine)

    def on_turn_start(self, turn_number, message, **kwargs):
        # Bookkeeping — increment turn counter in shared KV

    def get_tool_schemas(self):
        # memory_search, memory_grep, memory_read, mark_document

    def handle_tool_call(self, name, args, **kwargs):
        # Route to search.py / store

    def sync_turn(self, user_content, assistant_content, **kwargs):
        # FTS indexing happens on archive writes; no-op here
```

### Framework changes in hermes-agent

Autobio needs four small framework changes upstream. All are
backward-compatible: the built-in compressor and any third-party
engine inherit no-op defaults and are unaffected. Full diffs in
`HERMES_PATCH.md`.

#### 1. `on_turn_end(messages)` — per-turn message-list visibility

The base `ContextEngine` previously had no hook that fires per-turn
with full message-list visibility. `compress()` only runs on
threshold; `update_from_response` runs per API call (including
mid-tool-loop iterations) and has no messages; `on_session_end` is
the wrong granularity.

Without a per-turn hook, two features break:

1. **Speculative timing.** Speculation should fire when the tail is
   growing toward threshold. That's a per-turn decision with access
   to the live list. Without the hook, speculation can only be
   scheduled from inside `compress()` itself — which by definition
   runs too late.
2. **Proactive CM check-ins.** CM's check-in decision depends on
   recent turn content. Without message-list visibility, CM would
   operate on counts alone, or observations would queue and surface
   lazily later.

We added `on_turn_end(messages)` to the ABC with a no-op default.
The call site in `run_agent.py` fires it after the tool loop exits
and the final response is committed, guarded by try/except.

#### 2. `on_llm_call(api_kwargs, response, *, error=None)` — per-call audit

Without this hook, the only LLM calls landing in the archive's
`llm_calls` table are plugin-internal ones (summarizer, merge, CM
responses). The main agent's actual prompts — which is precisely
where memory-context rendering matters most — have nowhere to land
for inspection.

The hook fires once per API call (so N invocations per user turn for
an N-step tool chain) immediately after the response returns, with
the exact api_kwargs that went to the provider. The autobio engine
records each call as `purpose="main"` and handles three response
shapes: OpenAI-compatible (`response.choices[0].message.content`),
Anthropic-native (`response.content[N].text` blocks, skipping
non-text blocks like thinking/tool_use), and streaming-reconstructed
(SimpleNamespace mimicking the non-streaming shape upstream in the
runner). Streaming and direct-Anthropic paths are covered by
SimpleNamespace-based shape tests rather than MagicMock to catch
missing-attr assumptions.

#### 3. Engine-driven preflight via `should_compress_preflight(messages)`

The runner's preflight-compress check at session-resume time used to
be gated on a length guard:

```python
if (self.compression_enabled
    and len(messages) > protect_first_n + protect_last_n + 1):
    ...token threshold check...
```

For autobio transfer-in (session resumed with seeded memory in the
archive but only 1–2 short messages in `conversation_history`), the
length guard skipped the block entirely and the engine never got a
chance to render its memory recall pairs into conversation_history
before the first API call. Result: the agent's first turn after
transfer-in saw no memory at all.

Fix: runner consults `engine.should_compress_preflight(messages)`
alongside the length guard and bypasses both guards when the engine
fires. The base ContextEngine returns False by default (preserves
existing behavior). Autobio returns True whenever any L1 exists in
the archive — guarantees the first preflight after resume renders
recall pairs into the live list.

#### 4. System-prompt tuning knobs

The default Hermes system prompt carries a "You are Hermes Agent…"
identity layer plus tool-guidance blocks (`MEMORY_GUIDANCE`,
`SESSION_SEARCH_GUIDANCE`, `SKILLS_GUIDANCE`). For autobio
deployments — especially when transferring in an agent whose
identity should come from its accumulated memory — those defaults
fight the intent. The `MEMORY_GUIDANCE` block in particular
references a memory subsystem that autobio replaces.

Three additions to `_build_system_prompt` and `agent.*` config:

- `agent.skip_default_identity: bool` — skip the SOUL.md /
  DEFAULT_AGENT_IDENTITY block. When set, the runner also passes
  `skip_soul=True` to `build_context_files_prompt` so SOUL.md
  doesn't sneak back in via the context-files path.
- `agent.memory_guidance: str` — empty string suppresses the
  default memory guidance block; non-empty replaces verbatim;
  unset keeps default.
- `agent.session_search_guidance: str` — same shape as above.
- `agent.skills_guidance: str` — same shape as above.

All gated on the corresponding tool being loaded. Empty-string
suppression is the autobio-recommended default for the three
guidance blocks.

---

## Configuration

Defaults that match `plugins/autobio/config.py` as of v1. Anything not
specified in `~/.hermes/config.yaml` falls back to these values.

```yaml
context:
  engine: autobio

memory:
  provider: autobio

# --- agent-level system-prompt knobs (read by run_agent.py) ---
# These live under ``agent:``, NOT ``autobio:``, because they apply to
# the main agent's prompt assembly regardless of which context engine
# is active. They become essential for autobio because the default
# identity + memory-tool guidance assumes a memory subsystem autobio
# replaces.
agent:
  skip_default_identity: true   # skip "You are Hermes Agent…" + SOUL.md
  memory_guidance: ""           # suppress (autobio handles memory)
  session_search_guidance: ""   # suppress (autobio's archive search supersedes)
  skills_guidance: ""           # suppress / customize per deployment

# --- autobio-specific config ---
autobio:
  compression:
    # Weighted-token budget the picker accumulates per L1 chunk. Lower
    # values produce more, smaller, denser L1s (faster L2/L3 hierarchy
    # formation, more granular memory recall); higher values produce
    # fewer, coarser L1s. 6k weighted tokens lands ~10 L1s per
    # 250k-token tool-heavy session.
    chunk_tokens: 6000
    summary_target_tokens: 2500
    merge_threshold: 6
    max_levels: 3
    speculative: true
    speculative_lead_tokens: 20000
    summarize_retries: 3              # retry empty summarizer output
    # Optional: cap summarizer max_tokens directly. When unset, autobio
    # uses summary_target_tokens * 1.4. Reasoning models that consume
    # hidden completion tokens (k2.6, o1) may need a larger cap.
    # max_tokens_override: 6000

    # Weighted chunk sizing. Not all tokens cost the same in memory
    # terms: code/console output is recoverable from disk (CM can
    # re-read or grep), so it gets a discount; user/assistant text is
    # irreplaceable and counts at full weight. The chunk picker
    # accumulates WEIGHTED tokens up to ``chunk_tokens``.
    #
    # Weights apply ONLY to chunk-budget accounting. Threshold (when
    # to fire compress) and auto-doc detection stay raw — they're
    # about real model context limits.
    weights:
      default: 1.0
      user: 1.0
      assistant: 1.0
      system: 1.0
      tool_default: 0.5             # most tools (search, MCP, web fetch)
      tool_by_name:
        # Code reads / console output: cheap to recover.
        Read: 0.1
        Bash: 0.1
        Grep: 0.1
        Glob: 0.1
        terminal: 0.1
        read_file: 0.1
        # Document reads (extension-tagged): full content matters.
        # ``Read:ext`` form is set at archive time by tool_naming.
        # enrich_tool_name and takes precedence over the bare name.
        "Read:md": 0.5
        "Read:rst": 0.5
        "Read:txt": 0.5
        "Read:org": 0.5
        # Web — partial recovery.
        WebFetch: 0.5
        WebSearch: 0.5

  cm:
    # Bare ``claude-sonnet-4-6`` works on direct Anthropic API; the
    # provider-aware resolver namespaces it as
    # ``anthropic/claude-sonnet-4-6`` for OpenRouter automatically.
    model: "claude-sonnet-4-6"
    context_length: 1000000
    checkin_interval: 50
    forgetting_confirmation: false
    contradiction_detection: false    # deferred to v2
    proactive_surfacing: false        # deferred to v2
    defer_during_tool_chains: true

  search:
    max_results: 6
    min_score: 0.0
    # Embeddings deferred — FTS5 + grep only for v1.

  tool_pruning:
    default:
      full_retention_tokens: 30000
      summary_min_chars: 200
      argument_truncate_chars: 500
    per_tool:
      terminal:       { full_retention_tokens: 50000 }
      read_file:      { full_retention_tokens: 15000 }
      memory_search:  { full_retention_tokens: 10000 }
      write_file:     { argument_truncate_chars: 200 }
    dedup: true

  documents:
    auto_detect: true
    auto_threshold_tokens: 50000
    auto_style: chronological
    eligibility: size_pressure        # or 'manual'
    snap_boundary_window_tokens: 500
    exclude_tools: []                 # tool names that never auto-mark

  store:
    subdir: memory/autobio            # relative to hermes_home
```

### Config orphans (declared but not read)

The DEFAULTS dict carries a `tail` block (`default_tokens`,
`min_tokens`, `max_tokens`) reserved for an adaptive-tail feature
that didn't ship in v1. Currently no code path reads these keys; the
tail boundary is computed from the picker's `protect_last_tokens`
parameter inside compression. Setting them in user config has no
effect. Slated for cleanup or wiring in v2.

---

## Design decisions — why not otherwise

Reasoning preserved for future readers. Each of these corresponds to
a false start or rejected alternative we worked through.

### Why not: minted UUIDs as message IDs

We use SHA-256 truncated to 16 hex chars of (role, content,
tool_call_id, tool_calls). Hashes survive list renumbering, auto-dedup
identical messages, and compose naturally for chunk-level
fingerprints. Minted UUIDs would need assignment plumbing at ingress
and provide no upside.

One pitfall: hash the **archived** form (full, immutable), not the
live-list form. Tool-result pruning mutates the live list; if we
hashed current state, every prune would shift the hash and break
references.

### Why not: include tail-after-chunk in the summarizer prompt

Early drafts included everything from the tail except the chunk, in
chronological order, with the chunk appended at the end with the
marker. This felt like "show the model its current state, then the
memory it's about to form." Problem: it shows the model the **future**
of the chunk. The summary gets written with retrospective framing —
"I read the log, *which turned out to be* relevant to the bug I found
next" — inheriting knowledge the instance couldn't have had at the
as-of moment.

Fixed by strict as-of ordering: nothing after the chunk in the prompt,
ever. The chunk is the newest thing the summarizer sees.

### Why not: CM can compress arbitrary middle ranges

Earlier drafts imagined `CM.compress(turn_range)` as an op. This
would mean summaries could be created out of chronological order and
then we'd need to filter "summaries that existed at time T" when
building the as-of prompt for a later chunk. Complex and easy to get
wrong.

We cut it. Compression is strictly chronological (oldest uncompressed
chunk first). CM's role is **display resolution**, not compression
schedule. The ops CM has — `UNCOMPRESS`, `REFOLD`, `MARK_DOCUMENT` —
control what layer renders where in the live view. The archive's
growth pattern is driven by tail pressure, not CM whim.

### Why not: pin blocks compression (write lock)

If a pin blocks summarization itself, pinned content accumulates
unarchived until the pin releases. At release time, we'd have a huge
burst of summarization work with no pre-computation possible. Worse,
if the pin is on content we need to search, search can't find it
because there's no summary indexed.

Treating pin as a display lock — summarize eagerly, fold on release —
makes the system work smoothly. Archive growth is uniform; fold
decisions are reversible; search always has something to find.

### Why not: separate `documents` table

The natural instinct is that a document is a different kind of thing
from a pin. It isn't. A document is a pin with (a) a style tag saying
how to fold when forced, (b) a source tag (auto-detected from a
tool result, or manually marked), and (c) a progress marker as
chunks retire. None of those need a new table. They're columns on
pins.

Keeping documents as pin+kind also unifies lookup logic. The chunk
picker asks "is this chunk inside a protected range?" once and
branches on kind. Separate tables would require two queries and
duplicated overlap logic.

### Why not: combined_hash cache for speculation

First-draft speculation used a hash of the **entire** summarizer input
(prior summaries + tail + chunk) as the cache key. Rationale: if
anything changes, invalidate.

Problem: rolling back a message in the tail (nowhere near the chunk)
invalidated the cache. More fundamentally, it conflated "the summary
is valid" with "the context around the summary hasn't changed." The
summary is a *memory of a chunk*. Its validity depends only on the
chunk content, not on what surrounds it. Archive-first means the
memory stands on its own once written.

Replaced with source_hash dedup (hash of chunk member hashes only).
The summaries table IS the cache. Rollback that doesn't touch the
chunk doesn't invalidate anything. Rollback that touches the chunk
produces a new source_hash → fresh summary written; the old one stays
harmlessly in the archive as a memory of a slice that no longer
appears in the live view.

### Why not: embeddings in v1

CM is already an LLM-in-the-loop reading the dashboard. All L3
summaries fit in CM's context. L2s mostly fit. CM can do hierarchical
traversal (L3 → L2 → L1) by reading summary content and deciding
what's relevant, which is strictly better than cosine similarity on
embeddings because CM can reason about the question.

FTS5 BM25 covers keyword search. Grep covers exact-match recall for
code, errors, URLs. The semantic-search niche — "find the discussion
about the *general concept*" even when exact terms don't match — is
small, and when it matters, CM can paraphrase the query as a search
op.

Embeddings add: a provider dependency, API cost per message, a
fallback chain, an LRU cache, chunking config, and a vector store
that doesn't scale past ~100k entries without ANN indexes (sqlite-vec
is brute force). All of that for marginal recall improvement CM can
mostly provide via reasoning.

Punted to v2 with a clear integration point (the `MemoryProvider`
search methods) so adding them later doesn't require architectural
change.

### Why not: fused compress + fold (original design)

The original `compress_to_l1` function did both archive write and
live-view mutation atomically. Problem: pinned content can't be
summarized in the background without also forcibly folding it, which
violates the pin.

Splitting `summarize_chunk` (archive) from `fold_chunk` (display)
unlocks:
- Background summarization of anything (pinned, unpinned, docs)
- Fold decisions gated cleanly on pin state + pressure
- Speculation simplified — worker just writes to archive; fold happens
  later in compress_step
- `unfold` becomes an obvious inverse

The refactor was straightforward once we realized pins-as-display-locks.

### Why not: speculative_cache table

Original cache lived in its own table keyed by the (now-abandoned)
combined_hash. After the source_hash refactor, the summaries table
itself answers "is this chunk summarized?" via
`find_summary_by_source_hash`. The extra table is redundant and slated
for removal.

### Why not: write tail-after-chunk into the prompt but mark it "future"

Considered marking tail-after-chunk as `[FUTURE — do not reference]`
and leaving it in the prompt for positional stability. Rejected:
attention is attention. Instructing the model to ignore content that
is nonetheless in context is weaker than just not putting it there.

---

## V1 scope vs. deferred

**Built and tested:**

- Plugin skeleton (both ContextEngine and MemoryProvider halves)
- SQLite archive with content-hash message IDs, FTS5
  (content hash includes `reasoning_details` so thinking-only turns
  don't collapse to identical hashes)
- Three-pass tool-result pruning with per-tool retention
- `summarize_chunk` / `fold_chunk` / `unfold` / `compress_step`
- L1→L2→L3 merging at configurable thresholds
  (L3 path verified end-to-end with synthetic merge_threshold=2)
- Speculative summarization via archive-dedup
- Recall-pair dialogue rendering (CM-voice user header + agent-voice
  body) — replaces the earlier single-marker design
- Balance rule: render walks `messages.compressed_into` directly so
  every L1 emits a recall pair regardless of merge state above it
- `UNCOMPRESS` / `REFOLD` ops with persistence in the
  `unfolded_summaries` KV (state survives across renders until
  explicitly refolded)
- `MARK_DOCUMENT` / `LIST_DOCUMENTS` / `UNMARK_DOCUMENT` ops with
  document folding (chronological + monolithic styles)
- CM as tool (`cm_manage_context`) with op parser
- Dashboard with summary previews (L1 row includes merged-into-L2 L1s
  so CM can act on anything currently in the agent's prompt)
- Proactive CM check-ins (in-dialog via `on_turn_end`)
- Provider-aware CM model resolution (bare `claude-sonnet-4-6`
  namespaced to `anthropic/claude-sonnet-4-6` on OpenRouter)
- FTS5 + grep search
- Four framework changes upstream in hermes-agent:
  `on_turn_end`, `on_llm_call`, `should_compress_preflight`,
  system-prompt tuning knobs
- Main-agent prompt audit trail in `llm_calls` table with
  `purpose=main` (records exact prompt + response on every API call;
  handles OpenAI, Anthropic-native, and streaming-reconstructed
  shapes)
- Transfer-in workflow: resume a session with seeded memory; first
  preflight renders recall pairs into conversation_history before
  the first API call
- Crash resilience: orphan-summary recovery via
  `find_summary_by_source_hash` content-addressed dedup. Verified
  with both synthetic injection and a real subprocess SIGKILL
  mid-compress
- 40 passing mock tests covering the above

**Deferred to v2:**

- **Adaptive resolution** per region (when budget tight, fall back
  from L1-fidelity rendering to L2/L3 representations — what the
  balance rule's "lower orbitals first" wording leaves room for).
  This is the structural prerequisite for re-exposing summary-level
  PIN/UNPIN.
- **Summary-level PIN / UNPIN** ops — the `summaries.pinned` column
  and op handlers exist but are gated off in v1 (op handler returns
  an explanatory error). Re-introduced once adaptive resolution
  makes the flag observable.
- **Embeddings** for semantic search — interface point reserved
- **Contradiction detection** (depends on embeddings)
- **Proactive surfacing** (topic pattern tracking)
- **Forgetting confirmation**
- **Catastrophic recovery** from archive (the archive is durable
  under SQLite WAL; what's deferred is a higher-level workflow for
  reconstructing a session entirely from archive when the live
  conversation_history is lost)
- **Recursive CM spawning** for deep research across many layers
- **Multi-user / per-user topic tracking** in gateway sessions

---

## Open questions

1. **CM model cost**. Sonnet-class on every check-in is not cheap.
   Should CM use a smaller context window for routine check-ins and
   only load full context for complex ops? Mitigation: check-ins
   DEFER most of the time when things are quiet.

2. **CM memory of its own decisions**. CM forgets its past negotiations
   once they rotate out of the main instance's tail. The ops results
   persist in the layer store, but the reasoning behind decisions is
   lost. Could store a lightweight CM decision log in the KV table.

3. **Pin budget**. Unbounded pinning could prevent fold entirely.
   Should there be a max pin budget (e.g., 30% of tail)? CM could
   warn when approaching the limit.

4. **Multi-user sessions** (gateway). Different users in a group chat
   may have different memory relevance. Should CM maintain per-user
   topic tracking? v1 keeps it single-user-per-session via session_id
   scoping; multi-user is future work.

5. **Grep performance at scale**. After months of continuous operation
   the JSONL archive could be very large. Currently grep operates on
   summary markdown files (much smaller than raw messages). If we add
   grep-over-raw, we may need pre-filtering by date range or layer.

6. **Document boundary snapping at scale**. Current heuristic is
   regex-based (timestamps, headers, blank lines). Adequate for most
   structured text but imperfect. Could be upgraded to LLM-guided
   slicing if it matters for specific document types.
