"""Autobiographical memory system — shared core package.

Imported by both ``plugins/context_engine/autobio/`` (the ContextEngine
plugin) and ``plugins/memory/autobio/`` (the MemoryProvider plugin).
Not itself a discoverable plugin — lives under ``plugins/`` purely as
a convenient sibling package that both halves can import from.

The two halves share a single SQLite archive via a module-level store
singleton (see ``store.get_store``).
"""
