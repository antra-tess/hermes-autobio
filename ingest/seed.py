"""Archive seeding from an adapter stream.

Writes an adapter's ``IngestedMessage`` stream into an autobio session's
``messages`` table in chronological order. This is the production
migration path: take a long-lived instance's existing transcript, seed
the autobio archive, and start the new engine session pointing at that
archive.

This module does NOT compress on its own — that's the engine's job
once it's running. Seeding is purely the archive-write step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from ..ids import message_hash
from ..store import Store
from .base import IngestedMessage, TranscriptAdapter

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    inserted: int = 0
    skipped: int = 0          # duplicates by hash (already in archive)
    failed: int = 0
    sample_failures: List[str] = field(default_factory=list)
    last_turn: int = 0
    last_position: int = -1

    def summary_line(self) -> str:
        return (
            f"seeded {self.inserted} messages "
            f"({self.skipped} dup-skipped, {self.failed} failed); "
            f"last turn={self.last_turn} position={self.last_position}"
        )


def seed_archive(
    store: Store,
    source: TranscriptAdapter | Iterable[IngestedMessage],
    *,
    starting_position: int = 0,
    raise_on_failure: bool = False,
) -> SeedResult:
    """Write an adapter's stream into ``store`` in chronological order.

    Each message is hashed and upserted via ``store.upsert_message``,
    which is idempotent (INSERT OR IGNORE on hash). Re-running the seed
    on a partially-imported archive is safe — duplicates are reported
    in ``SeedResult.skipped``.

    Args:
        store: open autobio Store for the destination session.
        source: a ``TranscriptAdapter`` or any iterable of ``IngestedMessage``.
        starting_position: position offset for the first message. Use
            this when seeding multiple transcripts back-to-back into one
            session so positions stay monotonic.
        raise_on_failure: if True, the first upsert failure raises. If
            False (default), failures are logged at error level and the
            seed continues — partial migration beats no migration.

    Returns:
        SeedResult with insertion counts and the final turn/position.
    """
    result = SeedResult()
    if isinstance(source, TranscriptAdapter):
        stream = source.iter_messages()
    else:
        stream = iter(source)

    position = starting_position
    for msg in stream:
        try:
            openai_msg = msg.to_openai()
            h = message_hash(openai_msg)
            # Detect duplicates explicitly so we can report skipped count.
            existed = store.get_message(h) is not None
            store.upsert_message(
                h, openai_msg, turn=msg.turn, position=position,
            )
            if existed:
                result.skipped += 1
            else:
                result.inserted += 1
            result.last_turn = max(result.last_turn, msg.turn)
            result.last_position = position
            position += 1
        except Exception as e:
            result.failed += 1
            err = f"position={position} turn={msg.turn} role={msg.role}: {e}"
            logger.error("seed_archive: insert FAILED — %s", err)
            if len(result.sample_failures) < 5:
                result.sample_failures.append(err)
            if raise_on_failure:
                raise

    logger.info("seed_archive: %s", result.summary_line())
    return result
