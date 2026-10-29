# Changelog

## 0.5.0
Retours d'usage sur de vrais providers depuis le 0.4.0.
- **Recherche Web** (`ricochet.search`, `web_search`, `register_search_tool`, `python -m ricochet search`, `GET /api/search/status`) : Exa, Tavily et LangSearch derrière une interface, bascule sur un refus, quota suivi sur disque avec arrêt à 90 % du plafond gratuit, outil `search_web` pour les modèles. `docs/RECHERCHE_WEB.md` : les providers retenus et ceux écartés, avec la raison. Adaptateurs testés contre des réponses simulées uniquement.
- **Refus mieux classés** : 402 et limite par minute à zéro (offre non activée) = pause d'une heure ; 429 sur un plafond journalier = pause d'une heure au lieu d'une minute ; `x-ratelimit-reset-tokens` lu quand il n'y a pas de `Retry-After`.
- **Budget de jetons par modèle** lu dans les en-têtes des réponses réussies : un modèle qui n'a plus de quoi traiter la demande n'est pas appelé ; la limite annoncée s'apprend sans attendre un 413.
- **`read_json`** : lecture tolérante d'une réponse JSON (écrite deux fois, champ en trop, `null` parasite) sans jamais deviner une réponse coupée.
- **`find_personal_data` / `check_conversation`** : repère e-mail, téléphone, IBAN, carte, sécurité sociale (clés de contrôle) avant envoi à un free tier.
- Les blocs `<think>...</think>` ne sont plus rendus comme réponse. Gemini : chaîne `gemini-3.6-flash`, `3.5-flash`, `3.1-pro-preview`. Gonka passe après OpenRouter. `RICOCHET_ORDER` règle la tête de liste sans fichier. `httpx` devient une dépendance directe.
- 212 tests (aucun réseau).
- Hors périmètre : le client Gemini natif (appels d'outils et écriture en direct, `thoughtSignature`), qui suppose une boucle asynchrone et un format propre à Gemini ; Gemini reste appelé en format OpenAI, sans outils.

## 0.4.0
- **Lecture par lot** : `generate()` (appel isolé : bascule entre providers et entre modèles, jamais de réponse vide, délai réglable qui ne fait que s'allonger, `only` / `prefer`) et `run_parallel()` (travailleurs par provider, plafond de simultanés, écart entre départs, provider en pause sans travail, arrêt propre).
- **État par modèle** : un 429, un 413 ou une réponse vide n'écarte qu'un modèle ; le provider n'est en pause que si plus aucun de ses modèles ne répond. Le 413 apprend la limite du modèle.
- **Catalogue** : `extra_models`, `model_options` (couper le raisonnement des Nemotron : `enable_thinking: false`), `token_limits`, `max_concurrency`, `min_interval_s`, `env_required`, `env_aliases`, adresse à gabarit `{VARIABLE}`.
- **Providers** : NVIDIA mis à jour (ultra-550b, super-vl-120b, bolt-e4b), Groq (3 modèles, budget par modèle), Ollama, **Cloudflare Workers AI** et **Gonka** ajoutés.
- `python -m ricochet models <provider>` : quels modèles listés répondent vraiment. `docs/GUIDE.md` : pièges de la lecture par lot.
- 149 tests (aucun réseau).

## 0.3.0
- Providers configurables sans toucher au code (`providers.json`, validé) ; registre d'outils `@tool` ; plugins (`RICOCHET_PLUGIN`) ; prompt système via `set_system_prompt`.
- Nouvelle interface : fond beige, une ligne par provider (barre de quota + KPI), détail au clic.
- Le serveur refuse d'écouter hors de la machine locale sans `--allow-remote`.
- Suppression du code métier d'exemple (`rechercher_client`) ; ajout de `examples/`, CI, lint (ruff), 88+ tests.

## 0.2.0
- Paquet `ricochet`, tableau de bord web, providers NVIDIA NIM / Ollama Cloud / OpenRouter, statistiques et journal.

## 0.1.0
- Routeur multi-providers avec bascule sur quota et tool calling (fichier unique).
