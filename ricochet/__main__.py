"""
Ligne de commande :
  python -m ricochet                 → serveur web + tableau de bord (défaut)
  python -m ricochet serve --port N  → idem, port choisi
  python -m ricochet status          → état des providers dans le terminal
  python -m ricochet chat "bonjour"  → une question, une réponse
  python -m ricochet models nvidia   → quels modèles listés par le provider répondent VRAIMENT (diagnostic)
  python -m ricochet search "requête" → recherche Web par API gratuite (Exa, Tavily, LangSearch), quota suivi
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
    rows = [("PROVIDER", "ÉTAT", "PAUSE", "LATENCE", "OK/ERR", "QUOTA")]
    for p in router.quota_status().values():
        if not p["configured"]:
            rows.append((p["label"], "désactivé", "", "", "", f"clé absente ({p['env_key']})"))
            continue
        st = p["stats"]
        quota = " · ".join(
            f"{lim['label']} {lim['remaining']}" + (f"/{lim['limit']}" if lim["limit"] else "")
            for lim in p["limits"]
        ) or "non communiqué"
        rows.append((
            p["label"], p["status"], f"{p['cooldown_s']}s" if p["cooldown_s"] else "",
            f"{st['avg_latency_ms']} ms" if st["avg_latency_ms"] else "",
            f"{st['ok']}/{st['errors']}", quota,
        ))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows) + _fmt_search_status()


def _fmt_search_status() -> str:
    """Moteurs de recherche Web, sous les providers de modèles. Rien si aucune clé n'est configurée."""
    engines = [e for e in search.status()["providers"] if e["configured"]]
    if not engines:
        return ""
    lines = ["", "", "RECHERCHE WEB  ÉTAT          CONSOMMÉ"]
    for e in engines:
        state = "en pause" if e["paused"] else "disponible" if e["available"] else "quota atteint"
        quota = f"/{e['quota']:g}" if e["quota"] else ""
        lines.append(f"{e['label']:<14} {state:<13} {e['used']:g}{quota} {e['unit']}/{e['period']}")
    return "\n".join(lines)


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _serve(host: str, port: int, open_browser: bool, allow_remote: bool) -> int:
    import uvicorn

    from .server import create_app

    if host not in _LOCAL_HOSTS and not allow_remote:
        print(f"Refusé : écouter sur {host} expose le tableau de bord (sans authentification) au réseau.\n"
              "Ajoutez --allow-remote si c'est voulu, de préférence derrière un proxy authentifié.", file=sys.stderr)
        return 2
    url = f"http://{host}:{port}"
    print(f"Tableau de bord : {url}   (Ctrl+C pour arrêter)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="ricochet", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="serveur web + tableau de bord")
    serve.add_argument("--host", default="127.0.0.1", help="127.0.0.1 par défaut (local uniquement)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="ouvre le navigateur")
    serve.add_argument("--allow-remote", action="store_true", help="autorise un --host non local (sans authentification !)")
    sub.add_parser("status", help="état des providers")
    chat = sub.add_parser("chat", help="poser une question")
    chat.add_argument("message")
    models = sub.add_parser("models", help="sonder les modèles listés par un provider : lesquels répondent vraiment ?")
    models.add_argument("provider")
    models.add_argument("--all", action="store_true", help="inclure aussi embeddings, vision seule, parole…")
    models.add_argument("--timeout", type=float, default=40.0, help="secondes par modèle (40 par défaut)")
    find = sub.add_parser("search", help="recherche Web par API gratuite")
    find.add_argument("query")
    find.add_argument("-n", type=int, default=6, help="nombre de résultats (6 par défaut, 10 au plus)")
    find.add_argument("--content", action="store_true", help="demande le texte des pages quand le moteur le donne")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        load_plugin()
    except PluginError as e:
        print(f"Erreur : {e}", file=sys.stderr)
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
            print(f"Erreur : provider inconnu ou sans clé : {args.provider}", file=sys.stderr)
            return 1
        ok = [r for r in rows if r["status"] in ("ok", "raisonne")]
        print(f"{len(rows)} modèle(s) sondé(s) : {len(ok)} répondent, {len(rows) - len(ok)} non.")
        print()
        for r in rows:
            note = "  ← raisonne : couper le raisonnement (model_options) ou relever max_tokens" if r["status"] == "raisonne" else ""
            print(f"  {r['status']:<20} {r['latency_s']:>5.1f} s  {r['model']}{note}")
    elif args.cmd == "search":
        found = search.search(args.query, n=args.n, content=args.content)
        if found.status not in (search.SUCCESS, search.EMPTY):
            print(f"Erreur : {found.error or found.status}", file=sys.stderr)
            return 1
        for page in found.results:
            print(f"- {page['title']}\n  {page['url']}\n  {page['snippet']}")
        print(f"\n[{found.engine or '-'} · {len(found.results)} résultat(s) · coût {found.cost:g}]", file=sys.stderr)
    elif args.cmd == "chat":
        try:
            result = router.chat_detailed([], args.message)
        except router.LLMRouterError as e:
            print(f"Erreur : {e}", file=sys.stderr)
            return 1
        print(result.answer)
        print(f"\n[{result.provider} · {result.model} · {result.mode} · {result.latency_ms} ms]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
