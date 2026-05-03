"""Flask app for the autobio viewer.

Pragmatic: server-rendered shell pages, JSON endpoints for the
dynamic data. Vanilla JS in the browser for navigation. No client-side
framework.

Routes:

- ``GET /``                          — session picker
- ``GET /session/<sid>``             — main viewer page
- ``GET /api/sessions``              — JSON list of sessions
- ``GET /api/<sid>/messages``        — chronological messages
- ``GET /api/<sid>/summaries``       — full summary set with links
- ``GET /api/<sid>/cm_events``       — CM dialogue events
- ``GET /api/<sid>/llm_call/<id>``   — full LLM call (prompt + response)
- ``GET /api/<sid>/llm_calls``       — list LLM calls (filterable)

The viewer reads the same store the engine writes, so a session that
is currently running can be inspected live (refresh the page).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, abort, jsonify, render_template, request

from ..store import get_store
from . import data as data_layer

logger = logging.getLogger(__name__)


def create_app(hermes_home: str | Path) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["HERMES_HOME"] = str(Path(hermes_home).expanduser())

    @app.context_processor
    def _ctx() -> Dict[str, Any]:
        return {"hermes_home": app.config["HERMES_HOME"]}

    # -- pages -----------------------------------------------------------

    @app.route("/")
    def index():
        sessions = data_layer.list_sessions(app.config["HERMES_HOME"])
        return render_template("index.html", sessions=sessions)

    @app.route("/session/<sid>")
    def session(sid: str):
        # Validate session exists.
        store = _open_store(app, sid)
        return render_template(
            "session.html",
            session_id=sid,
            stats={
                "messages": store.message_count(),
                "summary_counts": store.summary_counts(),
            },
        )

    # -- JSON endpoints --------------------------------------------------

    @app.route("/api/sessions")
    def api_sessions():
        return jsonify(data_layer.list_sessions(app.config["HERMES_HOME"]))

    @app.route("/api/<sid>/messages")
    def api_messages(sid: str):
        store = _open_store(app, sid)
        msgs = data_layer.list_messages(store)
        # Annotate each with which L1 covers it (if any).
        summaries = data_layer.list_summaries(store)
        m2s = data_layer.message_to_summary_index(summaries)
        for m in msgs:
            m["l1_summary_id"] = m2s.get(m["hash"])
        return jsonify({"count": len(msgs), "messages": msgs})

    @app.route("/api/<sid>/summaries")
    def api_summaries(sid: str):
        store = _open_store(app, sid)
        summaries = data_layer.list_summaries(store)
        return jsonify({"count": len(summaries), "summaries": summaries})

    @app.route("/api/<sid>/cm_events")
    def api_cm_events(sid: str):
        store = _open_store(app, sid)
        events = data_layer.list_cm_events(store)
        return jsonify({"count": len(events), "events": events})

    @app.route("/api/<sid>/cm_event/<int:event_id>")
    def api_cm_event(sid: str, event_id: int):
        store = _open_store(app, sid)
        event = data_layer.get_cm_event(store, event_id)
        if event is None:
            abort(404)
        # Attach the LLM call that produced this event, if any.
        if event.get("llm_call_id"):
            event["llm_call"] = data_layer.get_llm_call(store, event["llm_call_id"])
        return jsonify(event)

    @app.route("/api/<sid>/llm_call/<int:call_id>")
    def api_llm_call(sid: str, call_id: int):
        store = _open_store(app, sid)
        call = data_layer.get_llm_call(store, call_id)
        if call is None:
            abort(404)
        return jsonify(call)

    @app.route("/api/<sid>/llm_calls")
    def api_llm_calls(sid: str):
        store = _open_store(app, sid)
        summary_id = request.args.get("summary_id")
        cm_event_id = request.args.get("cm_event_id", type=int)
        calls = data_layer.list_llm_calls(
            store, summary_id=summary_id, cm_event_id=cm_event_id,
        )
        return jsonify({"count": len(calls), "calls": calls})

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_store(app: Flask, sid: str):
    db = Path(app.config["HERMES_HOME"]) / "memory" / "autobio" / "store.db"
    if not db.exists():
        abort(404)
    return get_store(app.config["HERMES_HOME"], sid)
