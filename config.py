"""Config loading for the autobio plugin.

Reads the ``autobio:`` block from ``~/.hermes/config.yaml`` and applies
defaults for anything missing. Returns a nested dict with guaranteed
keys so the rest of the code never has to deal with ``.get(..., default)``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


DEFAULTS: Dict[str, Any] = {
    "tail": {
        "default_tokens": 100_000,
        "min_tokens": 50_000,
        "max_tokens": 150_000,
    },
    "compression": {
        "chunk_tokens": 10_000,
        "summary_target_tokens": 2_500,
        "merge_threshold": 6,
        "max_levels": 3,
        "speculative": True,
        "speculative_lead_tokens": 20_000,
        # Weighted chunk sizing: not all tokens cost the same in
        # memory terms. Code/console output is recoverable from disk
        # (CM can re-read or grep), so it gets a discount; user and
        # assistant text is irreplaceable and counts at full weight.
        # The chunk picker accumulates WEIGHTED tokens up to
        # ``chunk_tokens``, so a code-heavy region produces a single
        # coarser L1 instead of many small ones.
        #
        # Weights apply only to chunk-budget accounting. Threshold
        # (when to fire compress) and auto-doc detection stay raw —
        # they're about real model context limits.
        "weights": {
            "default": 1.0,
            "user": 1.0,
            "assistant": 1.0,
            "system": 1.0,
            "tool_default": 0.5,    # most tools (search, MCP, web fetch)
            "tool_by_name": {       # cheap-to-recover tool outputs
                "Read": 0.1, "Bash": 0.1, "Grep": 0.1, "Glob": 0.1,
                "ls": 0.1, "cat": 0.1, "head": 0.1, "tail": 0.1,
                "terminal": 0.1, "read_file": 0.1, "search_files": 0.1,
                "WebFetch": 0.5, "WebSearch": 0.5,
            },
        },
    },
    "cm": {
        "model": "claude-sonnet-4-6",
        "context_length": 1_000_000,
        "checkin_interval": 50,
        "forgetting_confirmation": False,
        "contradiction_detection": False,  # deferred to v2
        "proactive_surfacing": False,      # deferred to v2
        "defer_during_tool_chains": True,
    },
    "search": {
        # Embeddings deferred — FTS5 + grep only for v1.
        "max_results": 6,
        "min_score": 0.0,  # BM25 only, no min score filtering
    },
    "tool_pruning": {
        "default": {
            "full_retention_tokens": 30_000,
            "summary_min_chars": 200,
            "argument_truncate_chars": 500,
        },
        "per_tool": {
            "terminal": {"full_retention_tokens": 50_000},
            "read_file": {"full_retention_tokens": 15_000},
            "memory_search": {"full_retention_tokens": 10_000},
            "write_file": {"argument_truncate_chars": 200},
        },
        "dedup": True,
    },
    "store": {
        "subdir": "memory/autobio",  # relative to hermes_home
    },
    "documents": {
        "auto_detect": True,
        "auto_threshold_tokens": 50_000,
        "auto_style": "chronological",
        "eligibility": "size_pressure",      # or 'manual'
        "snap_boundary_window_tokens": 500,   # reserved for future sub-message splits
        "exclude_tools": [],                  # tool names that never auto-mark
    },
}


def load_autobio_config() -> Dict[str, Any]:
    """Read user config from hermes_cli.config and merge onto DEFAULTS.

    Safe to call even if hermes_cli is not importable — returns DEFAULTS
    unchanged in that case.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception as e:
        logger.debug("autobio: hermes_cli.config unavailable (%s) — using defaults", e)
        return copy.deepcopy(DEFAULTS)

    user_block = (config or {}).get("autobio") or {}
    return _deep_merge(copy.deepcopy(DEFAULTS), user_block)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` in-place and return base."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
