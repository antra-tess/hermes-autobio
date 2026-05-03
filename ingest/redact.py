"""Secret redaction for cross-provider transcripts.

When a transcript is ingested into the same provider that produced it
(e.g. Anthropic → Anthropic), keys, tokens, and signatures are
expected to round-trip and there's nothing to redact. When it's
ingested into a different provider (e.g. an Anthropic transcript fed
to OpenRouter for fuzz testing), credentials in the transcript
content would leak to the destination provider.

This module provides a redaction pass to apply BEFORE the messages
hit the destination engine. Replaces matched secrets with
``[REDACTED:<type>]`` so the message stays grammatically intact and
the model can still reason about "an API key was discussed here."

By design:

- The :class:`ClaudeCodeAdapter` does NOT redact. It stays faithful
  to the source. Production migration paths that stay within one
  provider need the original keys.
- The fuzz harness defaults to redacting (``HarnessConfig.redact_secrets=True``)
  because it routes through OpenRouter / K2.6.

Patterns are deliberately conservative — overzealous redaction would
mangle code samples in tool outputs. Add new patterns when a real
leak slips through; don't speculate.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # Anthropic API keys (sk-ant-api03-...)
    ("anthropic_api_key",
     re.compile(r"sk-ant-api03-[A-Za-z0-9_\-]{60,}")),
    # OpenRouter v1 keys (sk-or-v1-...)
    ("openrouter_api_key",
     re.compile(r"sk-or-v1-[A-Za-z0-9]{40,}")),
    # OpenAI project keys (sk-proj-...)
    ("openai_project_key",
     re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}")),
    # OpenAI legacy + anything similarly shaped — last so it doesn't
    # eat the more specific patterns above.
    ("openai_legacy_key",
     re.compile(r"\bsk-[A-Za-z0-9]{40,}\b")),
    # GitHub PATs and fine-grained tokens.
    ("github_pat",
     re.compile(r"ghp_[A-Za-z0-9]{36,}")),
    ("github_fine_grained",
     re.compile(r"github_pat_[A-Za-z0-9_]{60,}")),
    # AWS access keys (and their secret-key adjacency in env-style
    # assignments — caught by the assignment pattern below).
    ("aws_access_key",
     re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # JWTs (header.payload.signature, all base64url). Headers can be
    # ~17 chars for minimal {"alg":"HS256"}, so we keep the threshold
    # low — three dot-separated base64 segments starting with "eyJ" is
    # already specific enough that false positives are rare.
    ("jwt_token",
     re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
]


# Env-style assignments: redact the value side, keep the key name so
# the message stays informative ("API_KEY=[REDACTED:env_assignment]").
_ENV_ASSIGN = re.compile(
    r"(?P<lhs>(?:password|passwd|secret|token|api[_-]?key|"
    r"access[_-]?key|private[_-]?key)\s*[:=]\s*['\"]?)"
    r"(?P<value>[A-Za-z0-9_\-\./+=]{16,})",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """Apply all patterns to a string and return the redacted version.

    Idempotent: running on already-redacted text is safe (placeholders
    do not match the patterns).
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for name, pat in _PATTERNS:
        out = pat.sub(f"[REDACTED:{name}]", out)
    out = _ENV_ASSIGN.sub(
        lambda m: f"{m.group('lhs')}[REDACTED:env_assignment]",
        out,
    )
    return out


def redact_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``msg`` with string fields redacted in place.

    Walks ``content`` (string OR list of blocks) and ``tool_calls``
    (function arguments). Does not touch ``reasoning_details`` —
    Anthropic thinking blocks have signatures that would invalidate if
    we modified the text, and they aren't sent verbatim across
    providers anyway. They get dropped (or the message keeps them; we
    chose to keep — caller decides whether to forward).
    """
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = redact_text(content)
    elif isinstance(content, list):
        out["content"] = [_redact_block(b) for b in content]
    tcs = out.get("tool_calls")
    if isinstance(tcs, list):
        out["tool_calls"] = [_redact_tool_call(tc) for tc in tcs]
    return out


def _redact_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    out = dict(block)
    if isinstance(out.get("text"), str):
        out["text"] = redact_text(out["text"])
    return out


def _redact_tool_call(tc: Any) -> Any:
    if not isinstance(tc, dict):
        return tc
    out = dict(tc)
    fn = out.get("function")
    if isinstance(fn, dict):
        new_fn = dict(fn)
        args = new_fn.get("arguments")
        if isinstance(args, str):
            new_fn["arguments"] = redact_text(args)
        out["function"] = new_fn
    return out
