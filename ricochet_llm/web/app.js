"use strict";
(() => {
  const REFRESH_MS = 3000;

  // ───────── DOM : tout est inséré en texte, aucune donnée n'est interprétée comme du HTML ─────────
  const $ = (sel, root = document) => root.querySelector(sel);
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const shape = (kind, color) => h("i", { class: `shape ${kind}`, style: color ? `--c:${color}` : null });

  // ───────── Formatage ─────────
  const nf = new Intl.NumberFormat("fr-FR");
  const fmt = (n) => (n == null ? "—" : nf.format(n));
  const fmtMs = (ms) => (ms == null ? "—" : ms >= 1000 ? (ms / 1000).toFixed(1).replace(".", ",") + " s" : ms + " ms");
  const fmtTok = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1).replace(".", ",") + " M" : n >= 1e3 ? (n / 1e3).toFixed(1).replace(".", ",") + " k" : String(n));
  const fmtClock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString("fr-FR");

  const STATUS = {
    ok:       { label: "Disponible",       shape: "circle" },
    unknown:  { label: "Prêt, non sondé",  shape: "hollow" },
    limit:    { label: "Quota bas",        shape: "triangle" },
    cooldown: { label: "En pause",         shape: "diamond" },
    disabled: { label: "Clé absente",      shape: "ring" },
  };
  const KIND = {   // type d'événement → [forme, couleur, libellé court]
    ok:         ["circle", "var(--ink)", "ok"],
    rate_limit: ["diamond", "var(--warn)", "quota"],
    transient:  ["triangle", "var(--warn)", "réseau"],
    config:     ["cross", "var(--danger)", "config"],
    payment:    ["cross", "var(--danger)", "offre"],
    rejected:  ["triangle", "var(--warn)", "refusé"],
    empty:      ["hollow", "var(--muted)", "vide"],
    no_choices: ["diamond", "var(--warn)", "sans réponse"],
    info:       ["ring", "var(--muted)", "info"],
  };

  let data = null, fetchedAt = 0, inFlight = false, timer = null;
  const rows = new Map();
  const open = new Set();
  let rowOrder = "";
  const label = (name) => (data && data.providers[name] && data.providers[name].label) || name;

  // ───────── Toast ─────────
  let toastTimer = null;
  function toast(text, bad = false) {
    const el = $("#toast");
    el.textContent = text;
    el.className = "toast show" + (bad ? " bad" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = "toast"; }, 3600);
  }

  // ───────── Résumé ─────────
  function renderSummary(d) {
    const ps = d.order.map((n) => d.providers[n]).filter((p) => p.configured);
    const avail = ps.filter((p) => p.status !== "cooldown").length;
    const ok = ps.reduce((a, p) => a + p.stats.ok, 0);
    const err = ps.reduce((a, p) => a + p.stats.errors, 0);
    const lats = ps.map((p) => p.stats.avg_latency_ms).filter((x) => x != null);
    const lat = lats.length ? Math.round(lats.reduce((a, b) => a + b, 0) / lats.length) : null;
    const kpi = (k, v) => h("div", null, h("dt", null, k), h("dd", null, v));

    const main = d.next.provider
      ? h("div", { class: "main" }, d.next.label, h("small", null, `${d.next.model} · ${d.next.mode}`))
      : h("div", { class: "main bad", title: d.next.error || "" }, "Aucun provider disponible");
    $("#summary").replaceChildren(
      h("div", null, h("span", { class: "label" }, "Prochain appel"), main),
      h("dl", { class: "kpis" },
        kpi("Actifs", `${avail} / ${ps.length}`), kpi("Latence moy.", fmtMs(lat)),
        kpi("Appels", fmt(ok + err)), kpi("Erreurs", ok + err ? `${Math.round((err / (ok + err)) * 100)} %` : "—")));
  }

  // ───────── Lignes providers ─────────
  function mostConstrained(limits) {
    const usable = (limits || []).filter((l) => l.limit > 0 && l.remaining != null);
    if (!usable.length) return null;
    return usable.reduce((a, b) => (b.remaining / b.limit < a.remaining / a.limit ? b : a));
  }

  function buildRow(name) {
    const r = { shape: h("i", { class: "shape circle" }), name: h("div", { class: "name" }), state: h("div", { class: "state" }) };
    r.fill = h("i"); r.barL = h("span"); r.barR = h("span", { class: "num" });
    r.lat = h("dd", { class: "num" }); r.ok = h("dd", { class: "num" }); r.err = h("dd", { class: "num" });
    r.act = h("div", { class: "act" });
    r.detail = h("div", { class: "detail", hidden: true });
    r.kpis = h("dl", { class: "kpis" },
      h("div", null, h("dt", null, "Latence"), r.lat), h("div", null, h("dt", null, "OK"), r.ok), h("div", null, h("dt", null, "Err."), r.err));
    r.line = h("div", { class: "line", role: "button", tabindex: "0", "aria-expanded": "false" },
      h("div", { class: "id" }, r.shape, h("div", { style: "min-width:0" }, r.name, r.state)),
      h("div", { class: "bar" }, h("div", { class: "track" }, r.fill), h("div", { class: "barlabel" }, r.barL, r.barR)),
      r.kpis, r.act);
    const toggle = () => {
      const isOpen = !open.has(name);
      isOpen ? open.add(name) : open.delete(name);
      r.detail.hidden = !isOpen;
      r.line.setAttribute("aria-expanded", String(isOpen));
      if (isOpen) { r.sig = null; renderRow(name, data.providers[name]); }
    };
    r.line.addEventListener("click", toggle);
    r.line.addEventListener("keydown", (ev) => { if ((ev.key === "Enter" || ev.key === " ") && ev.target === r.line) { ev.preventDefault(); toggle(); } });
    r.el = h("article", { class: "row", "data-name": name }, r.line, r.detail);
    return r;
  }

  function detailContent(name, p) {
    if (!p.configured) {
      return [h("p", null, "Ajoutez la clé dans ", h("b", null, ".env"), " : ", h("code", null, `${p.env_key}=…`)),
        p.note ? h("p", null, p.note) : null,
        p.console_url ? h("p", null, h("a", { href: p.console_url, target: "_blank", rel: "noopener noreferrer" }, "Obtenir une clé gratuite ↗")) : null];
    }
    const item = (k, v) => (v == null ? null : h("div", null, h("dt", null, k), h("dd", null, v)));
    const exp = p.key_expires_at
      ? `${p.key_expires_at} (${p.key_expires_in_days < 0 ? "expirée" : "dans " + p.key_expires_in_days + " j"})` : null;
    const soon = p.key_expires_in_days != null && p.key_expires_in_days <= 30;
    return [
      h("dl", { class: "grid" },
        item("Modèle heavy", p.model_heavy), item("Modèle light", p.model_light),
        item("Tokens consommés", p.stats.tokens ? fmt(p.stats.tokens) : null),
        item("Appels / minute", fmt(p.stats.calls_last_minute)),
        (p.limits || []).map((l) => item(l.label, l.limit != null ? `${fmt(l.remaining)} / ${fmt(l.limit)}` : fmt(l.remaining))),
        item("Clé · expiration", exp), item("Note", p.note)),
      soon ? h("p", { class: "warnline" }, "La clé expire bientôt : pensez à la renouveler.") : null,
      p.cooldown_s ? h("p", { class: "warnline" }, "Provider en pause : il sera retenté automatiquement.") : null,
      p.last_error ? h("p", { class: "alert" }, p.last_error) : null,
      h("div", { class: "acts" },
        p.cooldown_s ? h("button", { class: "btn small", type: "button", onclick: () => resetProvider(name) }, "Reprendre maintenant") : null,
        p.console_url ? h("a", { href: p.console_url, target: "_blank", rel: "noopener noreferrer" }, "Console du provider ↗") : null),
    ];
  }

  function renderRow(name, p) {
    let r = rows.get(name);
    if (!r) { r = buildRow(name); rows.set(name, r); }
    const st = STATUS[p.status] || STATUS.unknown;
    r.el.dataset.status = p.status;
    r.shape.className = `shape ${st.shape}`;
    r.name.textContent = p.label;
    r.cooldownEnd = p.cooldown_s ? fetchedAt + p.cooldown_s * 1000 : 0;
    r.stateText = st.label;
    r.state.textContent = st.label;

    if (p.configured) {
      const lim = mostConstrained(p.limits);
      if (lim) {
        const pct = Math.max(0, Math.min(1, lim.remaining / lim.limit));
        r.fill.style.width = (pct * 100).toFixed(1) + "%";
        r.fill.dataset.level = pct < 0.1 ? "low" : pct < 0.35 ? "mid" : "";
        r.barL.textContent = lim.label;
        r.barR.textContent = `${fmt(lim.remaining)} / ${fmt(lim.limit)}`;
      } else {
        r.fill.style.width = "0%";
        r.barL.textContent = "Quota non communiqué";
        r.barR.textContent = `${p.stats.calls_last_minute} appel${p.stats.calls_last_minute > 1 ? "s" : ""}/min`;
      }
      r.lat.textContent = fmtMs(p.stats.avg_latency_ms);
      r.ok.textContent = fmt(p.stats.ok);
      r.err.textContent = fmt(p.stats.errors);
      r.kpis.style.visibility = "";
      if (!r.testBtn) {
        r.testBtn = h("button", { class: "btn small", type: "button", onclick: (ev) => { ev.stopPropagation(); pingProvider(name, ev.currentTarget); } }, "Tester");
      }
      r.act.replaceChildren(r.testBtn);
    } else {
      r.fill.style.width = "0%";
      r.barL.textContent = p.env_key;
      r.barR.textContent = "";
      r.kpis.style.visibility = "hidden";
      r.act.replaceChildren(h("span", { class: "btn small", style: "visibility:hidden" }, "Tester"));
    }

    if (open.has(name)) {
      r.detail.hidden = false;
      r.line.setAttribute("aria-expanded", "true");
      const content = detailContent(name, p);
      const sig = JSON.stringify(p, (k, v) => (k === "stats" || k === "limits_age_s" ? undefined : v)) + p.stats.tokens + p.stats.calls_last_minute;
      if (r.sig !== sig) { r.detail.replaceChildren(...content); r.sig = sig; }
    }
    tickRow(r);
  }

  function tickRow(r) {
    if (!r.cooldownEnd) return;
    const left = Math.max(0, Math.ceil((r.cooldownEnd - Date.now()) / 1000));
    r.state.textContent = left ? `${r.stateText} · reprise dans ${fmtClock(left)}` : `${r.stateText} · reprise imminente`;
  }

  function renderRows(d) {
    for (const name of d.order) renderRow(name, d.providers[name]);
    for (const name of [...rows.keys()]) if (!d.order.includes(name)) rows.delete(name);
    const key = d.order.join(",");
    if (key !== rowOrder) { $("#rows").replaceChildren(...d.order.map((n) => rows.get(n).el)); rowOrder = key; }
  }

  // ───────── Journal ─────────
  function renderEvents(d) {
    const ul = $("#events");
    if (!d.events.length) { ul.replaceChildren(h("li", { class: "none" }, "Aucun événement pour l'instant.")); return; }
    ul.replaceChildren(...d.events.map((e) => {
      const [kind, color] = KIND[e.kind] || KIND.info;
      return h("li", null, h("span", { class: "t num" }, fmtTime(e.t)), shape(kind, color), h("span", { class: "m" }, h("b", null, label(e.provider)), e.message));
    }));
  }

  // ───────── Rafraîchissement ─────────
  function setLive(on) {
    $("#live").dataset.state = on ? "on" : "off";
    $("#live .shape").className = "shape " + (on ? "circle" : "cross");
    if (!on) $("#live-text").textContent = "hors ligne";
  }
  async function refresh() {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) throw new Error(String(res.status));
      data = await res.json();
      fetchedAt = Date.now();
      renderSummary(data); renderRows(data); renderEvents(data); setLive(true);
    } catch (e) {
      setLive(false);
    } finally {
      inFlight = false;
    }
  }
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(async () => { if (!document.hidden) await refresh(); schedule(); }, REFRESH_MS);
  }
  setInterval(() => {
    for (const r of rows.values()) tickRow(r);
    if (fetchedAt && $("#live").dataset.state === "on") {
      const s = Math.round((Date.now() - fetchedAt) / 1000);
      $("#live-text").textContent = s < 2 ? "à jour" : `mis à jour il y a ${s} s`;
    }
  }, 1000);

  // ───────── Actions ─────────
  async function post(url, body) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : null });
    let json = null;
    try { json = await res.json(); } catch (e) { /* corps vide */ }
    return { ok: res.ok, json };
  }
  async function pingProvider(name, btn) {
    btn.disabled = true; btn.textContent = "Test…";
    try {
      const { json } = await post(`/api/providers/${encodeURIComponent(name)}/ping`);
      if (json && json.ok) toast(`${label(name)} répond en ${fmtMs(json.latency_ms)}`);
      else toast(`${label(name)} : échec${json && json.error ? " — " + json.error : ""}`.slice(0, 220), true);
    } catch (e) {
      toast(`${label(name)} : serveur injoignable`, true);
    }
    btn.disabled = false; btn.textContent = "Tester";
    await refresh();
  }
  async function resetProvider(name) {
    await post(`/api/providers/${encodeURIComponent(name)}/reset`);
    toast(`${label(name)} réactivé : il sera retenté au prochain appel`);
    await refresh();
  }

  // ───────── Console ─────────
  const newId = () => (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()) + Math.random().toString(16).slice(2);
  let sessionId = newId(), sending = false;
  const msgs = $("#msgs");

  const showHint = () => msgs.replaceChildren(h("p", { class: "hint" }, "Posez une question : la réponse indique quel provider a répondu, et les bascules éventuelles."));
  function addMessage(role, text, extra) {
    const hint = $(".hint", msgs);
    if (hint) hint.remove();
    const el = h("div", { class: `msg ${role}` }, h("div", { class: "bubble" }, text), extra || null);
    msgs.append(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }
  function metaFor(m) {
    const parts = [];
    for (const t of m.trail || []) {
      const [kind, color, short] = KIND[t.outcome] || KIND.info;
      parts.push(h("span", { class: "chip", title: t.detail || "" }, shape(kind, color), `${label(t.provider)} · ${short}`), "→");
    }
    parts.push(h("span", { class: "chip", title: m.model }, shape("circle", "var(--ink)"), `${label(m.provider)} · ${m.model.split("/").pop()}`));
    parts.push(h("span", null, `${m.mode} · ${fmtMs(m.latency_ms)}${m.tool_rounds ? ` · ${m.tool_rounds} outil${m.tool_rounds > 1 ? "s" : ""}` : ""}`));
    return h("div", { class: "meta" }, parts);
  }
  async function send(text) {
    sending = true; $("#send").disabled = true;
    addMessage("user", text);
    const pending = addMessage("assistant pending", "Routage en cours…");
    try {
      const { ok, json } = await post("/api/chat", { session_id: sessionId, message: text });
      pending.remove();
      if (ok && json && json.status === "ok") addMessage("assistant", json.response, metaFor(json.meta));
      else addMessage("error", (json && json.response) || "Erreur inconnue");
    } catch (e) {
      pending.remove();
      addMessage("error", "Serveur injoignable.");
    }
    sending = false; $("#send").disabled = false; $("#input").focus();
    refresh();
  }
  $("#composer").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const input = $("#input"), text = input.value.trim();
    if (!text || sending) return;
    input.value = "";
    send(text);
  });
  $("#input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#new-chat").addEventListener("click", async () => {
    try { await post("/api/reset", { session_id: sessionId }); } catch (e) { /* hors ligne : on repart quand même */ }
    sessionId = newId();
    showHint();
  });
  $("#refresh").addEventListener("click", refresh);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });

  showHint();
  refresh();
  schedule();
})();
