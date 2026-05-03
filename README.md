# hermes-autobio

Hierarchical autobiographical memory plugin for [Hermes Agent].
Treats conversation history as autobiographical experience: raw
messages are never discarded — they fold progressively into layered
summaries that preserve the subjective arc of the conversation. A
dedicated Context Manager (CM) agent mediates between the main
instance and the memory layers.

This is a **plugin**, not a standalone library. It registers as both
a `ContextEngine` and a `MemoryProvider` with a host Hermes Agent
process. Hermes-agent itself needs one small framework hook —
`ContextEngine.on_turn_end(messages)` — see [HERMES_PATCH.md].

## Layout

```
.                           ← submodule-mounts at hermes-agent/plugins/autobio/
├── __init__.py             ← the plugins.autobio package
├── engine.py               ← AutobioContextEngine (compression + CM)
├── provider.py             ← AutobioMemoryProvider (search tools)
├── compression.py          ← summarize / fold / unfold / compress_step / merge
├── store.py                ← SQLite archive with FTS5
├── render.py               ← live view as a projection of archive state
├── search.py               ← memory_search / memory_grep / memory_read
├── speculative.py          ← background summarizer
├── tool_pruning.py         ← three-pass tool-result pruning
├── cm.py                   ← Context Manager dialogue + dashboard
├── config.py               ← config schema + defaults
├── ids.py                  ← content-hash message IDs
├── ingest/                 ← Claude Code session importer + redaction
├── viewer/                 ← web viewer (FastAPI + HTML)
│
├── docs/
│   ├── hierarchical-autobiographical-memory.md   ← design spec
│   ├── projection-model-refactor.md              ← V1 refactor (shipped)
│   └── coordinate-cleanup-refactor.md            ← V1.1 refactor (shipped)
└── tests/
    ├── test_autobio_plugin.py
    ├── test_autobio_thorough.py
    ├── test_autobio_coordinate_cleanup.py
    ├── test_autobio_live.py
    └── autobio_fuzz/                             ← fuzz harness (real LLM)
```

## Install (as a hermes-agent submodule)

From a hermes-agent checkout:

```sh
git submodule add ../hermes-autobio plugins/autobio
git submodule update --init
```

That mounts this repo's contents at `hermes-agent/plugins/autobio/`,
which is where Hermes' plugin discovery expects to find them. Two
thin shim packages still live in hermes-agent:

```
hermes-agent/plugins/context_engine/autobio/__init__.py    # 10-line register()
hermes-agent/plugins/memory/autobio/__init__.py            # 10-line register()
```

## Run tests

From hermes-agent root, with this repo mounted as a submodule:

```sh
pytest plugins/autobio/tests/test_autobio_plugin.py \
       plugins/autobio/tests/test_autobio_thorough.py \
       plugins/autobio/tests/test_autobio_coordinate_cleanup.py
```

(Add the path to hermes-agent `pyproject.toml`'s `testpaths` if you
want them in the default discovery set.)

The fuzz harness runs against a real LLM and needs `OPENROUTER_API_KEY`:

```sh
OPENROUTER_API_KEY=... python -m plugins.autobio.tests.autobio_fuzz.run_fuzz \
    --only smoke
```

Modes: `smoke`, `curated`, `scripted`, `random`, `cm`, `replay`, `all`.

## Spec

The full design lives in `docs/hierarchical-autobiographical-memory.md`.
Read it. The four invariants up top are load-bearing:

1. The archive is authoritative; the live view is a projection.
2. Compression is always chronological and never destructive.
3. Pins are display locks, not write locks.
4. Summarization preserves the as-of moment.

Every later design decision falls out of those four.

[Hermes Agent]: https://github.com/NousResearch/hermes-agent
[HERMES_PATCH.md]: ./HERMES_PATCH.md
