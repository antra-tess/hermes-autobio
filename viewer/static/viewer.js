/* autobio viewer — vanilla JS frontend.
 *
 * Loads three datasets from the JSON API (messages, summaries, cm_events)
 * and renders them into three panes. Click a row to populate the
 * detail pane on the right.
 *
 * The viewer is intentionally read-only — no mutations, no auth, no
 * websockets. Refresh to pick up new data while a session is running.
 */

const SID = window.SESSION_ID;

const state = {
  messages: [],
  summaries: [],
  cmEvents: [],
  llmCalls: [],
  byHash: {},
  byId: {},
  selected: null, // {kind, id}
};

// -------------------------------------------------------------------
// Data loading
// -------------------------------------------------------------------

async function loadAll() {
  const [msgs, sums, cms] = await Promise.all([
    fetch(`/api/${SID}/messages`).then(r => r.json()),
    fetch(`/api/${SID}/summaries`).then(r => r.json()),
    fetch(`/api/${SID}/cm_events`).then(r => r.json()),
  ]);
  state.messages = msgs.messages || [];
  state.summaries = sums.summaries || [];
  state.cmEvents = cms.events || [];
  state.byHash = {};
  for (const m of state.messages) state.byHash[m.hash] = m;
  state.byId = {};
  for (const s of state.summaries) state.byId[s.id] = s;
  renderMessages();
  renderHierarchy();
  renderCmEvents();
  renderLlmCallsList();
}

// -------------------------------------------------------------------
// Render: chronological messages
// -------------------------------------------------------------------

function renderMessages() {
  const list = document.getElementById("messages-list");
  list.innerHTML = "";
  const hideSR = document.getElementById("hide-system-reminders").checked;
  for (const m of state.messages) {
    const content = stringContent(m.content);
    if (hideSR && (content.startsWith("<system-reminder>") ||
                   content.startsWith("<local-command-"))) continue;
    const li = document.createElement("li");
    li.className = `role-${m.role}`;
    li.dataset.hash = m.hash;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `
      <span>#${m.position}</span>
      <span>turn ${m.turn ?? "?"}</span>
      <span>${m.role}</span>
      ${m.l1_summary_id ? `<span class="l1-tag">${m.l1_summary_id}</span>` : ""}
    `;
    const preview = document.createElement("div");
    preview.className = "preview";
    preview.textContent = previewOf(content, 200);
    li.appendChild(meta);
    li.appendChild(preview);
    li.addEventListener("click", () => selectMessage(m.hash));
    list.appendChild(li);
  }
}

function selectMessage(hash) {
  state.selected = { kind: "message", id: hash };
  highlightSelection();
  const m = state.byHash[hash];
  const sid = m.l1_summary_id;
  const html = [];
  html.push(`<h3>Message <code>${hash.slice(0,12)}</code></h3>`);
  html.push(`<div class="meta-line">turn ${m.turn ?? "?"} · pos ${m.position} · role <strong>${m.role}</strong>${sid ? ` · folded into <a class="action" href="#" data-summary="${sid}">${sid}</a>` : ""}</div>`);
  html.push(`<section><h3>content</h3>${renderContentBlock(m.content)}</section>`);
  if (m.tool_calls && m.tool_calls.length) {
    html.push(`<section><h3>tool_calls</h3><pre>${escape(JSON.stringify(m.tool_calls, null, 2))}</pre></section>`);
  }
  if (m.tool_call_id) {
    html.push(`<section><h3>tool_call_id</h3><code>${m.tool_call_id}</code></section>`);
  }
  if (m.reasoning_details) {
    html.push(`<section><h3>reasoning_details (preserved thinking)</h3><pre>${escape(JSON.stringify(m.reasoning_details, null, 2))}</pre></section>`);
  }
  setDetail(html.join(""));
  attachActionHandlers();
}

// -------------------------------------------------------------------
// Render: memory hierarchy
// -------------------------------------------------------------------

function renderHierarchy() {
  const tree = document.getElementById("hierarchy-tree");
  tree.innerHTML = "";
  // L3 → its children L2s → their children L1s
  const l3s = state.summaries.filter(s => s.level === 3);
  const l2s = state.summaries.filter(s => s.level === 2);
  const l1s = state.summaries.filter(s => s.level === 1);

  // Index merged_into for traversal
  const childrenOf = {};
  for (const s of state.summaries) {
    if (s.merged_into) {
      (childrenOf[s.merged_into] = childrenOf[s.merged_into] || []).push(s);
    }
  }
  const orphanL2 = l2s.filter(s => !s.merged_into);
  const orphanL1 = l1s.filter(s => !s.merged_into);

  for (const l3 of l3s) {
    tree.appendChild(renderTreeNode(l3, "level-l3", childrenOf));
  }
  for (const l2 of orphanL2) {
    tree.appendChild(renderTreeNode(l2, "level-l2", childrenOf));
  }
  for (const l1 of orphanL1) {
    tree.appendChild(renderTreeNode(l1, "level-l1", childrenOf));
  }
}

function renderTreeNode(s, cls, childrenOf) {
  const li = document.createElement("li");
  li.className = `${cls}${s.merged_into ? " merged" : ""}`;
  li.dataset.id = s.id;
  const tags = (s.tags || []).join(", ");
  li.innerHTML = `
    <div>
      <span class="id">${s.id}</span>
      <span style="color:#888">${s.tokens}t · turns ${s.first_turn ?? "?"}-${s.last_turn ?? "?"}${tags ? ` · ${tags}` : ""}</span>
    </div>
    <span class="preview">${escape(previewOf(s.content, 140))}</span>
  `;
  li.addEventListener("click", (e) => {
    e.stopPropagation();
    selectSummary(s.id);
  });
  // Recurse into children
  const kids = childrenOf[s.id] || [];
  if (kids.length) {
    const ul = document.createElement("ul");
    ul.className = "tree";
    for (const k of kids) {
      const childCls = k.level === 2 ? "level-l2" : k.level === 1 ? "level-l1" : "level-msg";
      ul.appendChild(renderTreeNode(k, childCls, childrenOf));
    }
    li.appendChild(ul);
  } else if (s.level === 1) {
    // Leaf L1: show its source messages as children
    const ul = document.createElement("ul");
    ul.className = "tree";
    for (const h of s.source_hashes || []) {
      const m = state.byHash[h];
      const msgLi = document.createElement("li");
      msgLi.className = "level-msg";
      msgLi.dataset.hash = h;
      msgLi.innerHTML = `
        <div><span class="id">${h.slice(0,10)}</span> <span style="color:#888">${m ? m.role : "?"}</span></div>
        <span class="preview">${escape(previewOf(m ? stringContent(m.content) : "<missing>", 120))}</span>
      `;
      msgLi.addEventListener("click", (e) => {
        e.stopPropagation();
        selectMessage(h);
      });
      ul.appendChild(msgLi);
    }
    li.appendChild(ul);
  }
  return li;
}

function selectSummary(id) {
  state.selected = { kind: "summary", id };
  highlightSelection();
  const s = state.byId[id];
  if (!s) return;
  const html = [];
  const lvlBadge = `<span class="badge l${s.level}">L${s.level}</span>`;
  html.push(`<h3>${lvlBadge} ${s.id}</h3>`);
  html.push(`<div class="meta-line">${s.tokens} tokens · gen ${s.compression_generations} · turns ${s.first_turn ?? "?"}-${s.last_turn ?? "?"}${s.merged_into ? ` · merged into <a class="action" href="#" data-summary="${s.merged_into}">${s.merged_into}</a>` : ""}</div>`);
  if (s.tags && s.tags.length) html.push(`<div class="meta-line">tags: ${s.tags.join(", ")}</div>`);
  html.push(`<section><h3>content</h3><pre>${escape(s.content || "")}</pre></section>`);
  // Source hashes / children
  if (s.source_hashes && s.source_hashes.length) {
    html.push(`<section><h3>sources (${s.source_hashes.length})</h3><ul class="ops-list">`);
    for (const sh of s.source_hashes) {
      if (s.level === 1) {
        const m = state.byHash[sh];
        html.push(`<li class="op-line"><a class="action" href="#" data-message="${sh}">${sh.slice(0,12)}</a> <span style="color:#888">${m ? m.role : "?"}</span> ${escape(previewOf(m ? stringContent(m.content) : "", 80))}</li>`);
      } else {
        const child = state.byId[sh];
        html.push(`<li class="op-line"><a class="action" href="#" data-summary="${sh}">${sh}</a> ${child ? `<span style="color:#888">${child.tokens}t</span>` : ""}</li>`);
      }
    }
    html.push("</ul></section>");
  }
  // LLM calls that produced this summary
  html.push(`<section><h3>LLM calls (loading...)</h3><div id="summary-llm-calls">…</div></section>`);
  setDetail(html.join(""));
  attachActionHandlers();
  loadSummaryLlmCalls(id);
}

async function loadSummaryLlmCalls(summaryId) {
  const res = await fetch(`/api/${SID}/llm_calls?summary_id=${encodeURIComponent(summaryId)}`).then(r => r.json());
  const target = document.getElementById("summary-llm-calls");
  if (!target) return;
  if (!res.calls || !res.calls.length) {
    target.innerHTML = `<p class="hint">No LLM calls recorded. (Were summaries written before the llm_calls table existed?)</p>`;
    return;
  }
  const html = [];
  for (const c of res.calls) {
    html.push(`<div style="margin: 8px 0;">`);
    html.push(`<a class="action" href="#" data-llm-call="${c.id}">call #${c.id} · ${c.purpose} · ${c.model || "?"}</a>`);
    if (c.error) html.push(` <span class="badge error">error</span>`);
    if (c.response_finish && c.response_finish !== "stop") {
      html.push(` <span class="badge">finish=${c.response_finish}</span>`);
    }
    if (c.prompt_tokens) html.push(` <span class="badge">in=${c.prompt_tokens}t</span>`);
    if (c.completion_tokens) html.push(` <span class="badge">out=${c.completion_tokens}t</span>`);
    html.push(`</div>`);
  }
  target.innerHTML = html.join("");
  attachActionHandlers();
}

// -------------------------------------------------------------------
// Render: CM events
// -------------------------------------------------------------------

function renderCmEvents() {
  const list = document.getElementById("cm-events-list");
  list.innerHTML = "";
  if (!state.cmEvents.length) {
    list.innerHTML = `<li class="hint" style="cursor:default">no CM events recorded</li>`;
    return;
  }
  for (const e of state.cmEvents) {
    const li = document.createElement("li");
    li.dataset.id = e.id;
    const dec = e.decision || (e.kind === "tool_call" ? "answer" : "?");
    li.innerHTML = `
      <div class="meta">
        <span>#${e.id}</span>
        <span>turn ${e.turn ?? "?"}</span>
        <span>${e.kind}</span>
        <span class="decision-${dec}">${dec}</span>
      </div>
      <div class="preview">${escape(previewOf(e.reply_text || e.raw_reply || "", 160))}</div>
    `;
    li.addEventListener("click", () => selectCmEvent(e.id));
    list.appendChild(li);
  }
}

async function selectCmEvent(id) {
  state.selected = { kind: "cm_event", id };
  highlightSelection();
  const event = await fetch(`/api/${SID}/cm_event/${id}`).then(r => r.json());
  const html = [];
  html.push(`<h3>CM event #${event.id} · ${event.kind}${event.decision ? ` → ${event.decision}` : ""}</h3>`);
  html.push(`<div class="meta-line">turn ${event.turn ?? "?"} · ${event.created_at}</div>`);
  html.push(`<section><h3>request</h3><pre>${escape(event.request || "")}</pre></section>`);
  if (event.dashboard) {
    html.push(`<section><h3>dashboard CM saw</h3><pre>${escape(JSON.stringify(event.dashboard, null, 2))}</pre></section>`);
  }
  html.push(`<section><h3>reply (parsed text)</h3><pre>${escape(event.reply_text || "")}</pre></section>`);
  html.push(`<section><h3>raw reply</h3><pre>${escape(event.raw_reply || "")}</pre></section>`);
  if (event.ops && event.ops.length) {
    html.push(`<section><h3>ops</h3><ul class="ops-list">`);
    for (const [op, result] of event.ops) {
      html.push(`<li><div class="op-line">▸ ${escape(op)}</div><div class="op-result">${escape(result)}</div></li>`);
    }
    html.push(`</ul></section>`);
  }
  if (event.llm_call) {
    html.push(`<section><h3>LLM call ${event.llm_call.id}</h3>`);
    html.push(`<a class="action" href="#" data-llm-call="${event.llm_call.id}">view full prompt → response</a>`);
    html.push(`</section>`);
  }
  setDetail(html.join(""));
  attachActionHandlers();
}

// -------------------------------------------------------------------
// Render: full LLM call (the prompt rendered as conversation)
// -------------------------------------------------------------------

async function renderLlmCallsList() {
  const list = document.getElementById("llm-calls-list");
  const res = await fetch(`/api/${SID}/llm_calls`).then(r => r.json());
  state.llmCalls = res.calls || [];
  list.innerHTML = "";
  if (!state.llmCalls.length) {
    list.innerHTML = `<li class="hint" style="cursor:default">no LLM calls recorded</li>`;
    return;
  }
  for (const c of state.llmCalls) {
    const li = document.createElement("li");
    li.dataset.id = c.id;
    if (c.error) li.classList.add("error");
    li.innerHTML = `
      <div class="meta">
        <span>#${c.id}</span>
        <span>${c.purpose}</span>
        <span>${c.model || ""}</span>
        ${c.summary_id ? `<span class="l1-tag">${c.summary_id}</span>` : ""}
        ${c.error ? `<span class="badge error">error</span>` : ""}
      </div>
      <div class="preview">${escape(previewOf(c.response_text || "(empty)", 160))}</div>
    `;
    li.addEventListener("click", () => selectLlmCall(c.id));
    list.appendChild(li);
  }
}

async function selectLlmCall(id) {
  state.selected = { kind: "llm_call", id };
  highlightSelection();
  const c = await fetch(`/api/${SID}/llm_call/${id}`).then(r => r.json());
  const html = [];
  html.push(`<h3>LLM call #${c.id} · ${c.purpose}</h3>`);
  html.push(`<div class="meta-line">${c.created_at} · model <code>${c.model || "?"}</code> · finish <code>${c.response_finish || "?"}</code>${c.prompt_tokens ? ` · ${c.prompt_tokens}+${c.completion_tokens}t` : ""}${c.summary_id ? ` · produced <a class="action" href="#" data-summary="${c.summary_id}">${c.summary_id}</a>` : ""}${c.cm_event_id ? ` · drove CM event <a class="action" href="#" data-cm-event="${c.cm_event_id}">#${c.cm_event_id}</a>` : ""}</div>`);
  if (c.error) html.push(`<section><span class="badge error">error</span> <code>${escape(c.error)}</code></section>`);
  // Prompt rendered as conversation
  html.push(`<section><h3>prompt (${(c.prompt || []).length} turns)</h3>`);
  html.push(`<ol class="conv-turns">`);
  for (const turn of (c.prompt || [])) {
    html.push(renderConvTurn(turn));
  }
  html.push(`</ol></section>`);
  // Response
  html.push(`<section><h3>response</h3>`);
  html.push(`<ol class="conv-turns"><li class="conv-turn role-assistant"><span class="role">assistant</span><div class="body">${escape(c.response_text || "(empty)")}</div></li></ol>`);
  html.push(`</section>`);
  setDetail(html.join(""));
  attachActionHandlers();
}

function renderConvTurn(turn) {
  const role = turn.role || "?";
  const content = turn.content;
  let body;
  if (typeof content === "string") {
    body = escape(content);
  } else if (Array.isArray(content)) {
    body = content.map(b => {
      if (b && b.type === "text") return escape(b.text || "");
      if (b && b.type === "image") return `<em>[image]</em>`;
      return `<pre>${escape(JSON.stringify(b))}</pre>`;
    }).join("\n");
  } else {
    body = escape(JSON.stringify(content));
  }
  let tcs = "";
  if (turn.tool_calls && turn.tool_calls.length) {
    tcs = turn.tool_calls.map(tc =>
      `<div class="tool-call">→ ${escape(((tc.function || {}).name) || "?")}(${escape(((tc.function || {}).arguments) || "")})</div>`
    ).join("");
  }
  return `<li class="conv-turn role-${role}"><span class="role">${role}${turn.tool_call_id ? ` · ${turn.tool_call_id}` : ""}</span><div class="body">${body}</div>${tcs}</li>`;
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

function setDetail(html) {
  document.getElementById("detail-body").innerHTML = html;
}

function highlightSelection() {
  // Clear all selections
  document.querySelectorAll(".messages-list .selected, .tree .selected, .cm-events-list .selected").forEach(el => el.classList.remove("selected"));
  if (!state.selected) return;
  const sel = state.selected;
  let el = null;
  if (sel.kind === "message") {
    el = document.querySelector(`.messages-list li[data-hash="${cssEscape(sel.id)}"]`);
  } else if (sel.kind === "summary") {
    el = document.querySelector(`.tree li[data-id="${cssEscape(sel.id)}"]`);
  } else if (sel.kind === "cm_event") {
    el = document.querySelector(`.cm-events-list li[data-id="${sel.id}"]`);
  }
  if (el) el.classList.add("selected");
}

function attachActionHandlers() {
  document.querySelectorAll("#detail-body a.action").forEach(a => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const sid = a.dataset.summary;
      const mid = a.dataset.message;
      const lid = a.dataset.llmCall;
      const cid = a.dataset.cmEvent;
      if (sid) selectSummary(sid);
      else if (mid) selectMessage(mid);
      else if (lid) selectLlmCall(parseInt(lid, 10));
      else if (cid) selectCmEvent(parseInt(cid, 10));
    });
  });
}

function stringContent(c) {
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c.map(b => (b && b.type === "text") ? b.text : `[${b && b.type}]`).join(" ");
  }
  return c == null ? "" : String(c);
}

function previewOf(s, n) {
  if (!s) return "";
  s = String(s).replace(/\s+/g, " ");
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function escape(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function cssEscape(s) {
  return String(s).replace(/"/g, '\\"');
}

// -------------------------------------------------------------------
// Wire up tabs + refresh
// -------------------------------------------------------------------

document.querySelectorAll(".tabs .tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tabs .tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById(`tab-${t.dataset.tab}`).classList.add("active");
  });
});

document.getElementById("hide-system-reminders").addEventListener("change", renderMessages);
document.getElementById("refresh-btn").addEventListener("click", loadAll);

loadAll();
