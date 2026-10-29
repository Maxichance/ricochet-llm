"""
Command line:
  python -m ricochet_llm                 → web server + dashboard (default)
  python -m ricochet_llm serve --port N  → same, with a chosen port
  python -m ricochet_llm status          → provider status in the terminal
  python -m ricochet_llm chat "hello"    → one question, one answer
  python -m ricochet_llm models nvidia   → which models listed by the provider REALLY answer (diagnostic)
  python -m ricochet_llm search "query"  → web search through free APIs (Exa, Tavily, LangSearch), quota tracked
"""
import argparse
import logging
import sys
import threading
import webbrowser

from . import router, search
from .plugins import PluginError, load_plugin


def _fmt_status() -> str:
    router.refresh_probes(force=True)
    rows = [("PROVIDER", "STATUS", "PAUSE", "LATENCY", "OK/ERR", "QUOTA")]
    for p in router.quota_status().values():
        if not p["configured"]:
            rows.append((p["label"], "disabled", "", "", "", f"key missing ({p['env_key']})"))
            continue
        st = p["stats"]
        quota = " · ".join(
            f"{lim['label']} {lim['remaining']}" + (f"/{lim['limit']}" if lim["limit"] else "")
            for lim in p["limits"]
        ) or "not reported"
        rows.append((
            p["label"], p["status"], f"{p['cooldown_s']}s" if p["cooldown_s"] else "",
            f"{st['avg_latency_ms']} ms" if st["avg_latency_ms"] else "",
            f"{st['ok']}/{st['errors']}", quota,
        ))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows) + _fmt_search_status()


def _fmt_search_status() -> str:
    """Web search engines, listed below the model providers. Nothing if no key is configured."""
    engines = [e for e in search.status()["providers"] if e["configured"]]
    if not engines:
        return ""
    lines = ["", "", "WEB SEARCH     STATUS        USED"]
    for e in engines:
        state = "paused" if e["paused"] else "available" if e["available"] else "quota reached"
        quota = f"/{e['quota']:g}" if e["quota"] else ""
        lines.append(f"{e['label']:<14} {state:<13} {e['used']:g}{quota} {e['unit']}/{e['period']}")
    return "\n".join(lines)


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _serve(host: str, port: int, open_browser: bool, allow_remote: bool) -> int:
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        print("The dashboard needs extra dependencies: pip install \"ricochet-llm[server]\"", file=sys.stderr)
        return 1

    if host not in _LOCAL_HOSTS and not allow_remote:
        print(f"Refused: listening on {host} exposes the dashboard (no authentication) to the network.\n"
              "Add --allow-remote if that is intended, preferably behind an authenticated proxy.", file=sys.stderr)
        return 2
    url = f"http://{host}:{port}"
    print(f"Dashboard: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0


def _reasoning_only(row: dict) -> bool:
    """Empty answer, but the model did write some reasoning (decided from the data, not from the status text)."""
    return bool(row.get("reasoning")) and not row.get("content_chars")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="ricochet-llm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logs")
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="web server + dashboard")
    serve.add_argument("--host", default="127.0.0.1", help="127.0.0.1 by default (local only)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="open the browser")
    serve.add_argument("--allow-remote", action="store_true", help="allow a non-local --host (no authentication!)")
    sub.add_parser("status", help="provider status")
    chat = sub.add_parser("chat", help="ask a question")
    chat.add_argument("message")
    models = sub.add_parser("models", help="probe the models listed by a provider: which ones really answer?")
    models.add_argument("provider")
    models.add_argument("--all", action="store_true", help="also include embeddings, vision-only, speech...")
    models.add_argument("--timeout", type=float, default=40.0, help="seconds per model (40 by default)")
    find = sub.add_parser("search", help="web search through a free API")
    find.add_argument("query")
    find.add_argument("-n", type=int, default=6, help="number of results (6 by default, 10 max)")
    find.add_argument("--content", action="store_true", help="ask for the page text when the engine provides it")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        load_plugin()
    except PluginError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.cmd in (None, "serve"):
        return _serve(getattr(args, "host", "127.0.0.1"), getattr(args, "port", 8765),
                      getattr(args, "open", False), getattr(args, "allow_remote", False))
    if args.cmd == "status":
        print(_fmt_status())
    elif args.cmd == "models":
        try:
            rows = router.probe_models(args.provider, timeout=args.timeout, include_all=args.all)
        except KeyError:
            print(f"Error: unknown provider or no key: {args.provider}", file=sys.stderr)
            return 1
        ok = [r for r in rows if r["status"] == "ok" or _reasoning_only(r)]
        print(f"{len(rows)} model(s) probed: {len(ok)} answer, {len(rows) - len(ok)} do not.")
        print()
        for r in rows:
            note = "  ← reasons: turn reasoning off (model_options) or raise max_tokens" if _reasoning_only(r) else ""
            print(f"  {r['status']:<20} {r['latency_s']:>5.1f} s  {r['model']}{note}")
    elif args.cmd == "search":
        found = search.search(args.query, n=args.n, content=args.content)
        if found.status not in (search.SUCCESS, search.EMPTY):
            print(f"Error: {found.error or found.status}", file=sys.stderr)
            return 1
        for page in found.results:
            print(f"- {page['title']}\n  {page['url']}\n  {page['snippet']}")
        print(f"\n[{found.engine or '-'} · {len(found.results)} result(s) · cost {found.cost:g}]", file=sys.stderr)
    elif args.cmd == "chat":
        try:
            result = router.chat_detailed([], args.message)
        except router.LLMRouterError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(result.answer)
        print(f"\n[{result.provider} · {result.model} · {result.mode} · {result.latency_ms} ms]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
