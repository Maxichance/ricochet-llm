"""
Catalogue des providers : données + chargement/validation. Ordre = préférence décroissante.
Un provider sans clé dans l'environnement est ignoré (affiché « désactivé »).

Personnaliser SANS toucher au code : un fichier `providers.json` (dossier courant, ou chemin dans
$RICOCHET_PROVIDERS) — voir providers.example.json :
  { "providers": [
      {"name": "groq", "model_heavy": "autre-modele"},                       # modifie un provider existant
      {"name": "cerebras", "enabled": false},                               # retire un provider
      {"name": "monprovider", "env_key": "MONPROVIDER_API_KEY",
       "base_url": "https://api.exemple.com/v1", "model_heavy": "modele"}   # en ajoute un
    ],
    "order": ["monprovider", "groq"] }                                      # optionnel : providers à passer en tête
  (ou, sans fichier : RICOCHET_ORDER=monprovider,groq)

Champs d'un provider :
  name / label      identifiant technique / nom affiché
  env_key           variable d'environnement de la clé API
                    (optionnel : <env_key>_EXPIRES=AAAA-MM-JJ pour afficher l'expiration de la clé)
  base_url          endpoint OpenAI-compatible
  model_heavy/light modèle « lourd » (mode HEAVY) et « léger » (mode LIGHT, quand tout est bas)
  quota_header      en-tête de réponse donnant le quota restant qui pilote le routage (None si absent)
  quota_min         seuil sous lequel le provider est écarté du mode HEAVY
  supports_tools    False si l'endpoint ne gère pas le tool calling multi-tours
  timeout           timeout HTTP en secondes (court pour basculer vite si le provider se bloque)
  extra_models      modèles de repli du même provider, essayés dans l'ordre après model_heavy quand un modèle est
                    écarté (429, 413, 404, réponse vide). Un modèle écarté ne met PAS tout le provider en pause.
                    model_light n'en fait partie que s'il y est listé : il sert au mode LIGHT (quota bas).
  model_options     réglages propres à un modèle : {"nvidia/nemotron-3-ultra-550b-a55b":
                    {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}}.
                    Indispensable pour couper le raisonnement des modèles qui « réfléchissent » (voir GUIDE.md).
  token_limits      plus grande demande acceptée par modèle, quand le provider l'annonce (Groq : 8 000 jetons/min
                    et par modèle). Sert à ne pas appeler un modèle qui refusera (413). Complété par apprentissage.
  max_concurrency   requêtes simultanées au plus pour ce provider (lecture par lot, voir parallel.py)
  min_interval_s    secondes minimales entre deux départs de requête chez ce provider (6 req/min = 10 s)
  env_required      variables d'environnement SUPPLÉMENTAIRES exigées (ex. identifiant de compte Cloudflare)
  env_aliases       noms alternatifs acceptés pour une variable : {"CLOUDFLARE_API_KEY": ["CLOUDFLARE_API_TOKEN"]}
                    `base_url` peut contenir {NOM_DE_VARIABLE}, remplacé par sa valeur d'environnement.
  quota_labels      libellés lisibles des en-têtes x-ratelimit-remaining-<suffixe>
  probe             sonde dédiée pour lire le quota hors des en-têtes (voir router.py)
  console_url       où obtenir une clé gratuite
  note              particularité à afficher
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

REQUEST_TIMEOUT_S = 60
CONFIG_FILENAME = "providers.json"
CONFIG_ENV_VAR = "RICOCHET_PROVIDERS"
ORDER_ENV_VAR = "RICOCHET_ORDER"          # « groq,gemini » : providers à passer en tête, sans fichier (noms inconnus ignorés)


class ConfigError(ValueError):
    """Configuration des providers invalide."""


BUILTIN_SPECS = [
    {
        "name":           "groq",
        "label":          "Groq",
        "env_key":        "GROQ_API_KEY",
        "base_url":       "https://api.groq.com/openai/v1",
        "model_heavy":    "openai/gpt-oss-120b",
        "model_light":    "openai/gpt-oss-20b",
        # Un budget de jetons par minute PAR MODÈLE : plusieurs modèles multiplient les appels avant le premier 429,
        # et un 429 sur l'un ne dit rien des autres.
        "extra_models":   ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
        "token_limits":   {"openai/gpt-oss-120b": 8000, "qwen/qwen3.8-27b": 7000, "openai/gpt-oss-20b": 8000},
        "quota_header":   "x-ratelimit-remaining-tokens",
        "quota_min":      2_000,
        "timeout":        30,
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "quota_labels":   {"tokens": "tokens / min", "requests": "requêtes / jour"},
        "console_url":    "https://console.groq.com/keys",
    },
    {
        "name":           "nvidia",
        "label":          "NVIDIA NIM",
        "env_key":        "NVIDIANIM_API_KEY",
        "base_url":       "https://integrate.api.nvidia.com/v1",
        # Le catalogue annonce beaucoup plus de modèles que ceux qui répondent avec une clé d'essai (`python -m ricochet_llm
        # models nvidia` donne la liste réelle). Les Nemotron RAISONNENT : sans consigne contraire ils dépensent tout
        # `max_tokens` dans `reasoning_content` et rendent un `content` vide. `enable_thinking: false` les fait répondre
        # directement. Ordre : le plus capable d'abord, le petit modèle en dernier repli.
        "model_heavy":    "nvidia/nemotron-3-ultra-550b-a55b",
        "extra_models":   ["nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3.5-bolt-e4b"],
        "model_light":    "nvidia/nemotron-3.5-bolt-e4b",         # 4B, ne raisonne pas : aucun réglage
        "model_options": {
            "nvidia/nemotron-3-ultra-550b-a55b": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            "nvidia/nemotron-3.5-super-vl-120b-a12b": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        },
        "timeout":        20,     # chat : on bascule vite. Une lecture longue passe `timeout=60` à generate()
        # Une limite en requêtes PAR MINUTE porte sur le rythme de départ, pas sur la simultanéité. Une salve passe, un flux
        # soutenu atteint la limite : on règle donc un plafond de travailleurs ET un écart entre départs (2 s = 30/min au
        # plus). Au-delà, des 429 inoffensifs : le routeur bascule sur le modèle suivant.
        "max_concurrency": 6,
        "min_interval_s": 2.0,
        "console_url":    "https://build.nvidia.com",
        "note":           "Essai : développement et test seulement (licence), pas de plafond annoncé en en-tête",
    },
    {
        "name":           "ollama",
        "label":          "Ollama Cloud",
        "env_key":        "OLLAMACLOUD_API_KEY",
        "base_url":       "https://ollama.com/v1",
        "model_heavy":    "gpt-oss:120b",
        "model_light":    "gpt-oss:20b",
        "extra_models":   ["gpt-oss:20b", "nemotron-3-nano:30b"],   # offre gratuite : les autres modèles répondent 402 (plan Pro)
        "timeout":        45,
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "console_url":    "https://ollama.com/settings/keys",
        "note":           "Limites non publiées (session 5 h / semaine)",
    },
    {
        "name":           "cloudflare",
        "label":          "Cloudflare Workers AI",
        "env_key":        "CLOUDFLARE_API_KEY",
        "env_required":   ["CLOUDFLARE_ACCOUNT_ID"],
        "env_aliases":    {"CLOUDFLARE_API_KEY": ["CLOUDFLARE_API_TOKEN"],
                           "CLOUDFLARE_ACCOUNT_ID": ["CLOUDFLAIRE_ACCOUNT_ID", "CF_ACCOUNT_ID"]},
        "base_url":       "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        # Modèles retenus : ceux de l'offre gratuite qui appellent l'outil attendu. Les modèles d'un plan payant répondent
        # 403 ; certains petits modèles répondent en texte au lieu d'appeler un outil.
        "model_heavy":    "@cf/openai/gpt-oss-120b",
        "extra_models":   ["@cf/nvidia/nemotron-3-120b-a12b", "@cf/google/gemma-4-26b-a4b-it", "@cf/zai-org/glm-4.7-flash"],
        "model_light":    "@cf/zai-org/glm-4.7-flash",
        "timeout":        45,     # lent : prévoir un délai large pour une lecture longue
        "max_concurrency": 2,
        "min_interval_s": 2.0,
        "console_url":    "https://dash.cloudflare.com/profile/api-tokens",
        "note":           "Plafond journalier (neurones) ; jeton « Workers AI » ~53 caractères",
    },
    {
        "name":           "openrouter",
        "label":          "OpenRouter",
        "env_key":        "OPENROUTER_API_KEY",
        "base_url":       "https://openrouter.ai/api/v1",
        "model_heavy":    "nvidia/nemotron-3-ultra-550b-a55b:free",
        "model_light":    "google/gemma-4-26b-a4b-it:free",
        "timeout":        45,
        "probe":          "openrouter",
        "console_url":    "https://openrouter.ai/keys",
        "note":           "50 requêtes gratuites par jour",
    },
    {
        "name":           "gonka",
        "label":          "Gonka Broker",
        "env_key":        "GONKA_API_KEY",
        "base_url":       "https://proxy.gonkabroker.com/v1",
        # Du plus rapide au plus lent.
        "model_heavy":    "deepseek-ai/DeepSeek-V4-Flash-0731",
        "extra_models":   ["zai-org/GLM-5.3-Flash", "MiniMaxAI/MiniMax-M2.7"],
        "model_light":    "MiniMaxAI/MiniMax-M2.7",
        "timeout":        45,
        # Offre gratuite : 6 requêtes par minute et 1 000 000 de jetons par MOIS : à réserver au chat, pas aux lectures
        # par lot.
        "max_concurrency": 1,
        "min_interval_s": 11.0,
        "console_url":    "https://gonkabroker.com",
        "note":           "6 requêtes/min, 1 M jetons/mois",
    },
    {
        "name":           "gemini",
        "label":          "Google Gemini",
        "env_key":        "GEMINI_API_KEY",
        "base_url":       "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Chaîne de repli : un modèle qui expire ou dont le quota du jour est épuisé est écarté, le suivant prend le relais
        # (`gemini-3.1-pro-preview` est le plus capable mais son quota gratuit par jour est très court).
        "model_heavy":    "gemini-3.6-flash",
        "extra_models":   ["gemini-3.5-flash", "gemini-3.1-pro-preview"],
        "model_light":    "gemini-flash-lite-latest",
        "supports_tools": False,
        "timeout":        45,
        "console_url":    "https://aistudio.google.com/apikey",
        "note":           "Sans tool calling (thought_signature) ; 20 requêtes/jour/modèle en gratuit",
    },
    {
        "name":           "mistral",
        "label":          "Mistral",
        "env_key":        "MISTRAL_API_KEY",
        "base_url":       "https://api.mistral.ai/v1",
        "model_heavy":    "mistral-small-latest",
        "quota_header":   "x-ratelimit-remaining-tokens-minute",
        "quota_min":      500,
        "timeout":        30,
        "quota_labels":   {"tokens-minute": "tokens / min", "req-minute": "requêtes / min"},
        "console_url":    "https://console.mistral.ai/api-keys",
        "note":           "Offre gratuite à vérifier : limite à zéro sans plan activé, mis en pause 1 h au premier refus",
    },
    {
        "name":           "cerebras",
        "label":          "Cerebras",
        "env_key":        "CEREBRAS_API_KEY",
        "base_url":       "https://api.cerebras.ai/v1",
        "model_heavy":    "llama3.1-8b",
        "quota_header":   "x-ratelimit-remaining-tokens-day",
        "quota_min":      5_000,
        "timeout":        30,
        "quota_labels":   {"tokens-day": "tokens / jour", "requests-day": "requêtes / jour", "req-minute": "requêtes / min"},
        "console_url":    "https://cloud.cerebras.ai",
        "note":           "Plus de free tier permanent (essai avec carte)",
    },
]

# ───────── validation ─────────

_DEFAULTS = {
    "quota_header": None, "quota_min": 1, "supports_tools": True, "timeout": REQUEST_TIMEOUT_S,
    "quota_labels": None, "probe": None, "console_url": None, "note": None,
    "extra_models": None, "model_options": None, "token_limits": None,
    "max_concurrency": 2, "min_interval_s": 0.0, "env_required": None, "env_aliases": None,
}
_MODEL_OPTION_KEYS = {"extra_body"}
_REQUIRED = ("name", "env_key", "base_url", "model_heavy")
_ALLOWED = set(_REQUIRED) | set(_DEFAULTS) | {"label", "model_light"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalize(spec: dict, where: str) -> dict:
    """Valide un provider et complète les valeurs par défaut. Lève ConfigError avec un message actionnable."""
    unknown = sorted(set(spec) - _ALLOWED)
    if unknown:
        raise ConfigError(f"{where} : champ(s) inconnu(s) {unknown}. Champs permis : {sorted(_ALLOWED | {'enabled'})}")
    missing = [k for k in _REQUIRED if not spec.get(k)]
    if missing:
        raise ConfigError(f"{where} : champ(s) obligatoire(s) manquant(s) : {missing}")
    out = {**_DEFAULTS, **spec}
    for key in ("name", "env_key", "base_url", "model_heavy"):
        if not isinstance(out[key], str):
            raise ConfigError(f"{where} : « {key} » doit être une chaîne")
    if not _NAME_RE.match(out["name"]):
        raise ConfigError(f"{where} : « name » doit être en minuscules (a-z, 0-9, - et _) : {out['name']!r}")
    if not out["base_url"].startswith(("http://", "https://")):
        raise ConfigError(f"{where} : « base_url » doit commencer par http:// ou https://")
    out["label"] = out.get("label") or out["name"]
    out["model_light"] = out.get("model_light") or out["model_heavy"]
    if not isinstance(out["quota_min"], int) or isinstance(out["quota_min"], bool) or out["quota_min"] < 0:
        raise ConfigError(f"{where} : « quota_min » doit être un entier ≥ 0")
    if not isinstance(out["supports_tools"], bool):
        raise ConfigError(f"{where} : « supports_tools » doit être true ou false")
    if isinstance(out["timeout"], bool) or not isinstance(out["timeout"], (int, float)) or out["timeout"] <= 0:
        raise ConfigError(f"{where} : « timeout » doit être un nombre de secondes > 0")
    if out["quota_header"] is not None and not isinstance(out["quota_header"], str):
        raise ConfigError(f"{where} : « quota_header » doit être une chaîne ou null")
    labels = out["quota_labels"]
    if labels is not None and not (isinstance(labels, dict) and all(isinstance(v, str) for v in labels.values())):
        raise ConfigError(f"{where} : « quota_labels » doit être un objet {{suffixe: libellé}}")
    out["quota_labels"] = dict(labels or {})

    extra = out["extra_models"]
    if extra is not None and not (isinstance(extra, list) and all(isinstance(m, str) and m for m in extra)):
        raise ConfigError(f"{where} : « extra_models » doit être une liste de noms de modèles")
    out["extra_models"] = list(extra or [])

    options = out["model_options"]
    if options is not None:
        if not (isinstance(options, dict) and all(isinstance(m, str) and isinstance(o, dict) for m, o in options.items())):
            raise ConfigError(f"{where} : « model_options » doit être un objet {{modèle: {{réglages}}}}")
        for model, opt in options.items():
            unknown_opt = sorted(set(opt) - _MODEL_OPTION_KEYS)
            if unknown_opt:
                raise ConfigError(f"{where} : « model_options » → {model} : réglage(s) inconnu(s) {unknown_opt} "
                                  f"(permis : {sorted(_MODEL_OPTION_KEYS)})")
            if "extra_body" in opt and not isinstance(opt["extra_body"], dict):
                raise ConfigError(f"{where} : « model_options » → {model} → extra_body doit être un objet")
    out["model_options"] = {m: dict(o) for m, o in (options or {}).items()}

    limits = out["token_limits"]
    if limits is not None and not (isinstance(limits, dict) and all(
            isinstance(m, str) and isinstance(n, int) and not isinstance(n, bool) and n > 0 for m, n in limits.items())):
        raise ConfigError(f"{where} : « token_limits » doit être un objet {{modèle: entier > 0}}")
    out["token_limits"] = dict(limits or {})

    mc = out["max_concurrency"]
    if isinstance(mc, bool) or not isinstance(mc, int) or not 1 <= mc <= 30:
        raise ConfigError(f"{where} : « max_concurrency » doit être un entier entre 1 et 30")
    mi = out["min_interval_s"]
    if isinstance(mi, bool) or not isinstance(mi, (int, float)) or mi < 0:
        raise ConfigError(f"{where} : « min_interval_s » doit être un nombre de secondes ≥ 0")

    required = out["env_required"]
    if required is not None and not (isinstance(required, list) and all(isinstance(v, str) and v for v in required)):
        raise ConfigError(f"{where} : « env_required » doit être une liste de noms de variables")
    out["env_required"] = list(required or [])
    aliases = out["env_aliases"]
    if aliases is not None and not (isinstance(aliases, dict) and all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(a, str) for a in v) for k, v in aliases.items())):
        raise ConfigError(f"{where} : « env_aliases » doit être un objet {{variable: [noms alternatifs]}}")
    out["env_aliases"] = {k: list(v) for k, v in (aliases or {}).items()}
    return out


def _config_path(path: Optional[str]) -> Optional[Path]:
    explicit = path or os.getenv(CONFIG_ENV_VAR)
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ConfigError(f"Fichier de providers introuvable : {p}")
        return p
    default = Path.cwd() / CONFIG_FILENAME
    return default if default.is_file() else None


def _env_first(order: List[str]) -> List[str]:
    """`$RICOCHET_ORDER` passe ses providers en tête (utile en déploiement : on change l'ordre sans toucher au fichier).
    Un nom inconnu est ignoré : une faute de frappe ne doit pas empêcher le routeur de démarrer."""
    wanted = [n.strip().lower() for n in (os.getenv(ORDER_ENV_VAR) or "").split(",")]
    return list(dict.fromkeys([n for n in wanted if n in order] + order))


def load_specs(path: Optional[str] = None) -> List[dict]:
    """Catalogue intégré, puis surcharges de providers.json (si présent). Retourne des specs validées et ordonnées."""
    specs = {s["name"]: _normalize(dict(s), f"catalogue intégré → {s['name']}") for s in BUILTIN_SPECS}
    order = [s["name"] for s in BUILTIN_SPECS]

    cfg = _config_path(path)
    if cfg is None:
        return [specs[n] for n in _env_first(order)]
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{cfg.name} illisible : {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("providers", []), list):
        raise ConfigError(f"{cfg.name} : attendu {{\"providers\": [ … ]}}")

    for i, entry in enumerate(data.get("providers", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ConfigError(f"{cfg.name} : l'entrée n°{i + 1} doit être un objet avec un champ « name »")
        name = entry["name"]
        if entry.get("enabled", True) is False:
            specs.pop(name, None)
            if name in order:
                order.remove(name)
            continue
        merged = {**specs.get(name, {}), **{k: v for k, v in entry.items() if k != "enabled"}}
        specs[name] = _normalize(merged, f"{cfg.name} → {name}")
        if name not in order:
            order.append(name)

    first = data.get("order", [])
    if not isinstance(first, list) or any(n not in specs for n in first):
        unknown = [n for n in first if n not in specs] if isinstance(first, list) else first
        raise ConfigError(f"{cfg.name} : « order » doit lister des providers connus (inconnus : {unknown})")
    order = list(dict.fromkeys(first + order))
    return [specs[n] for n in _env_first(order)]


PROVIDER_SPECS = load_specs()
