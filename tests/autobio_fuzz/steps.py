"""Step types driven by the generators and consumed by the harness.

Each Step is a dataclass describing one mutation of the live state.
The harness interprets them; generators produce lists of them. Keeping
them reified (rather than procedural callbacks) means a scenario is a
data structure that can be seeded, replayed, serialized into the
report, and compared across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AddUser:
    content: str


@dataclass
class AddAssistant:
    content: str


@dataclass
class AddToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    # Synthetic tool_call_id — generator picks, harness passes through.
    call_id: str


@dataclass
class Rollback:
    """Drop the last ``n`` messages from the live list (excluding system)."""
    n: int


@dataclass
class MarkPin:
    """Mark a range in the CURRENT live list as a pin."""
    start_index: int
    end_index: int
    kind: str = "pin"      # pin | doc_chronological | doc_monolithic
    reason: str = "test"
    meta: Optional[Dict[str, Any]] = None


@dataclass
class UnmarkPin:
    # Matched by (start_index, end_index, kind) at resolve time.
    start_index: int
    end_index: int
    kind: str = "pin"


@dataclass
class TriggerCompress:
    """Force a compress() call regardless of threshold."""
    pass


@dataclass
class CallCM:
    """Invoke ``cm_manage_context`` with a specific request.

    The harness wires this through ``engine.handle_tool_call`` so the
    full CM path (dashboard build, prompt, op parsing, op execution)
    runs and gets recorded in the CM transcript.
    """
    request: str


@dataclass
class TriggerProactiveCheckin:
    """Force a proactive CM check-in regardless of interval/turn count.

    The harness clears the deferred_until / interval barriers, calls
    ``maybe_proactive_checkin``, and appends any INJECT result to the
    live list (mirroring on_turn_end). Used in CM scenarios where we
    want INJECT decisions, not DEFER.
    """
    pass


@dataclass
class Turn:
    """Meta-step: a single conversational turn that fires on_turn_end at the end.

    Holds an arbitrary list of sub-steps (AddUser, AddAssistant, AddToolCall,
    …) that get applied in order; after the sub-steps the harness invokes
    ``engine.on_turn_end(messages)`` as the real runner would. This lets
    scenarios model a whole turn as one structural unit and keeps
    speculative/proactive-CM timing aligned with production behavior.
    """
    substeps: List[Any] = field(default_factory=list)


Step = Any  # union of the above; Python structural typing is fine here
