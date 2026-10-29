# Ricochet

[![tests](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml)
[![licence MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Un routeur qui répartit vos requêtes entre plusieurs API LLM **gratuites** (Groq, NVIDIA NIM, Ollama Cloud, Cloudflare Workers AI, Gonka, OpenRouter, Gemini, Mistral…), bascule tout seul quand un quota est épuisé ou qu'un provider ou un modèle tombe, gère le tool calling, exécute des **traitements par lot en parallèle** sur tous les providers à la fois, et affiche l'état de chaque provider dans un tableau de bord.

## Démarrage

```bash
git clone https://github.com/Maxichance/ricochet-llm.git && cd ricochet-llm
pip install -e .                  # ou : pip install -r requirements.txt
cp .env.example .env              # Windows : copy .env.example .env ; renseignez au moins une clé (voir plus bas)
python -m ricochet serve --open
```

Le tableau de bord s'ouvre sur **http://127.0.0.1:8765**. Il écoute uniquement sur votre machine : l'écoute sur une autre adresse est refusée sans `--allow-remote`, car l'interface n'a pas d'authentification et ses boutons consomment des quotas.

| Commande | Effet |
|---|---|
| `python -m ricochet serve [--port N] [--open]` | serveur web + tableau de bord (commande par défaut) |
| `python -m ricochet status` | état des providers dans le terminal |
| `python -m ricochet chat "bonjour"` | une question, une réponse (+ provider utilisé) |
| `python -m ricochet models nvidia` | **quels modèles listés répondent vraiment** (la liste d'un provider gratuit ne le dit pas) |

## Le tableau de bord

Une ligne par provider : statut, **barre de quota** et trois KPI (latence, appels ok, erreurs). Un clic déplie le détail : modèles, toutes les jauges, tokens consommés, expiration de la clé, dernière erreur. **Tester** fait un appel minimal (clé valide ? latence ?) et lève une éventuelle pause ; **Reprendre** lève la pause sans appeler. La console permet d'essayer le routage en direct (elle montre quel provider a répondu et les bascules) ; le journal liste les erreurs récentes.

Les quotas ne sont affichés que quand le provider les communique (en-têtes `x-ratelimit-*`, ou `/key` pour OpenRouter) ; sinon la ligne le dit.

## Utiliser le routeur dans votre code

```python
from ricochet import chat_detailed, tool

@tool
def rechercher_client(email: str) -> dict:
    """Recherche un client par email.

    Args:
        email: Email du client
    """
    return {"email": email, "nom": "Dupont"}      # votre vrai code métier

history = []                                      # une liste par conversation
r = chat_detailed(history, "Trouve jean@exemple.fr")
print(r.answer, r.provider, r.model, r.trail)
```

Le schéma JSON de l'outil est déduit de la signature et de la docstring. Les erreurs d'un outil sont renvoyées au modèle au lieu de planter la conversation. `set_system_prompt("…")` définit le prompt système. Voir `examples/use_as_library.py`.

**Pour le serveur et le CLI**, mettez vos outils dans un fichier ou un module et pointez dessus : `RICOCHET_PLUGIN=examples/demo_tools.py` (voir cet exemple). Le plugin est du code exécuté tel quel : ne chargez que du code de confiance.

## Traiter un lot : plusieurs appels à la fois

```python
from ricochet import generate, run_parallel

def lire(document, provider):                       # exécutée dans un thread
    r = generate(prompt(document), max_tokens=8192, timeout=60, prefer=provider)
    sauvegarder(document, r.text, r.provider, r.model)   # au fil de l'eau : une interruption ne perd rien

bilan = run_parallel(documents, lire, limits={"nvidia": 10})   # {"processed", "errors", "remaining", "per_provider", …}
```

`generate` bascule entre providers **et entre modèles** (un 429 ou un 413 n'écarte qu'un modèle, pas tout le provider), ne renvoie jamais une réponse vide, et accepte un délai plus long que celui du chat. `run_parallel` donne à chaque provider ses propres travailleurs, avec un plafond de simultanés et un écart entre départs adaptés à ses limites. Les réglages adaptés à chaque provider (couper le raisonnement des Nemotron, limites de jetons par modèle de Groq, débit de Gonka…) sont dans le catalogue ; les pièges sont dans `docs/GUIDE.md`.

## Chercher sur le Web, lire un JSON imparfait, filtrer les données personnelles

```python
from ricochet import web_search, register_search_tool, read_json, find_personal_data

register_search_tool()                                 # l'outil `search_web` devient appelable par les modèles de chat()
pages = web_search("routeur llm gratuit", n=5, content=True).results     # Exa, Tavily, LangSearch : gratuits, quota suivi

data = read_json(generate(prompt, max_tokens=8192).text)   # dict, ou None : réponse écrite deux fois, `null` parasite, ```json...
if find_personal_data(message):                        # e-mail, téléphone, IBAN, carte, sécurité sociale (clés de contrôle)
    ...                                                # ne l'envoyez pas à un free tier
```

La recherche s'arrête à 90 % du quota gratuit de chaque moteur et bascule sur le suivant (`docs/RECHERCHE_WEB.md`). `read_json` ne répare que ce qui est sans ambiguïté et laisse illisible une réponse coupée ou qui boucle : on la relit, on ne la devine pas. Le contrôle du contenu reste à vous.

## Configurer les providers sans toucher au code

Copiez `providers.example.json` en `providers.json` (ou pointez dessus avec `RICOCHET_PROVIDERS`) pour modifier un modèle, retirer un provider, en ajouter un (tout endpoint compatible OpenAI) ou changer l'ordre de priorité (ou `RICOCHET_ORDER=groq,gemini` dans `.env`, sans fichier). Le fichier est validé : une faute de frappe donne une erreur explicite. Tous les champs sont documentés en tête de `ricochet/catalog.py`.

## Structure

```
ricochet/
  catalog.py     providers intégrés + chargement/validation de providers.json
  router.py      moteur : choix du provider et du modèle, bascule, état par modèle, `chat`, `generate`, `probe_models`
  parallel.py    exécuteur parallèle de traitements par lot (plafonds et écarts par provider)
  tools.py       registre d'outils (@tool)
  plugins.py     chargement de votre code métier
  cleaning.py    nettoyage des tool calls qui fuitent en texte, des blocs <think> (fonctions pures)
  search.py      recherche Web par API gratuite (Exa, Tavily, LangSearch), quota suivi sur disque
  jsonread.py    lecture tolérante de la réponse JSON d'un modèle
  privacy.py     repère les données personnelles évidentes avant envoi
  server.py      API web (FastAPI) : /api/status, /api/chat, /api/search/status, /api/providers/{nom}/ping …
  web/           tableau de bord (HTML/CSS/JS, sans dépendance)
  __main__.py    ligne de commande
examples/        plugin d'outils, usage en bibliothèque
tests/           tests avec de faux providers (aucun réseau, aucun quota)
docs/            guide technique, comportements des providers, recherche Web
```

Développement : `pip install -r requirements-dev.txt ruff`, puis `ruff check .` et `python -m unittest discover -s tests -v`.

## Clés gratuites

| Provider | Où l'obtenir |
|---|---|
| Groq | https://console.groq.com/keys |
| NVIDIA NIM | https://build.nvidia.com (clé d'essai, durée limitée) |
| Ollama Cloud | https://ollama.com/settings/keys |
| OpenRouter | https://openrouter.ai/keys |
| Gemini | https://aistudio.google.com/apikey |
| Mistral | https://console.mistral.ai/api-keys (vérifiez que l'offre gratuite est active) |
| Cloudflare Workers AI | https://dash.cloudflare.com/profile/api-tokens (jeton « Workers AI » **et** identifiant de compte) |
| Gonka | https://gonkabroker.com (6 requêtes/min, 1 M jetons/mois) |
| Exa (recherche Web) | https://dashboard.exa.ai (10 $ de crédit par mois, sans carte) |
| Tavily (recherche Web) | https://app.tavily.com (1 000 crédits par mois, sans carte) |
| LangSearch (recherche Web) | https://langsearch.com (allocation de jetons par jour) |

Un provider sans clé est ignoré. `NVIDIANIM_API_KEY_EXPIRES=AAAA-MM-JJ` dans `.env` affiche l'expiration d'une clé (alerte à J-30).

> **Les catalogues gratuits changent vite** (modèles retirés, limites modifiées) : testez un modèle par un vrai appel avant de le configurer, la liste `/models` d'un provider n'est pas fiable. Voir `docs/GUIDE.md`.
>
> **NVIDIA NIM** : la licence de l'offre gratuite couvre la recherche, le développement et le test ; la production relève d'une licence payante. Aucun en-tête de quota n'est renvoyé.
>
> **Confidentialité** : plusieurs free tiers journalisent ou utilisent vos prompts pour entraîner leurs modèles (Google hors UE, Mistral en free mode, OpenRouter, NVIDIA). N'y envoyez pas de données personnelles ou confidentielles sans avoir vérifié leurs conditions.

## Licence

[MIT](LICENSE). Sécurité et signalement d'une faille : [SECURITY.md](SECURITY.md).
