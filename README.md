<div align="center">

# Ricochet

**Un seul appel. Plusieurs API gratuites. Zéro quota épuisé.**

Le routeur LLM qui rebondit d'un provider gratuit à l'autre : quota atteint, modèle retiré, panne, réponse vide... votre code ne voit rien.

[![tests](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/Maxichance/ricochet-llm/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-1d3a8a.svg)](https://github.com/Maxichance/ricochet-llm/blob/main/pyproject.toml)
[![licence MIT](https://img.shields.io/badge/licence-MIT-1d3a8a.svg)](https://github.com/Maxichance/ricochet-llm/blob/main/LICENSE)
[![providers](https://img.shields.io/badge/providers-8%2B%20gratuits-1d3a8a.svg)](#les-providers)

[Démarrage](#démarrage) · [Fonctionnalités](#fonctionnalités) · [Dans votre code](#dans-votre-code) · [Providers](#les-providers) · [Configuration](#configuration) · [Documentation](#documentation)

<br>

<img src="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/dashboard.png" alt="Tableau de bord de Ricochet : état de chaque provider, quotas, console de test et journal" width="860">

<sub>Le tableau de bord (données de démonstration) : quota de chaque provider, pauses, console avec la bascule en direct, journal.</sub>

</div>

<br>

## Pourquoi

Les offres gratuites des API de LLM sont généreuses, mais chacune a ses limites : quota par minute ou par jour, modèles qui disparaissent, erreurs 429, réponses vides, délais qui explosent. Les enchaîner à la main, c'est du code fragile.

Ricochet met tout cela derrière **une seule interface**. Il choisit le provider et le modèle, surveille les quotas, écarte ce qui ne répond plus et réessaie ailleurs, sans jamais perdre la conversation.

```
 votre code ──►  chat()  ·  generate()  ·  run_parallel()  ·  web_search()
                                   │
                          ┌────────▼────────┐
                          │    Ricochet     │   429 · 402 · 5xx · réponse vide · quota bas
                          │  choisit · suit │──────────────────────┐
                          │  bascule        │                      │ rebondit : modèle suivant,
                          └────────┬────────┘                      │ puis provider suivant
        ┌───────┬────────┬─────────┼──────────┬─────────┐          │
      Groq   NVIDIA   Ollama   Cloudflare  OpenRouter  Gemini ... ◄┘
```

## Fonctionnalités

| | |
|---|---|
| **Bascule automatique** | Par provider **et par modèle** : un 429 n'écarte qu'un modèle, le provider ne se met en pause que si plus aucun de ses modèles ne répond. Les refus sont classés (quota, offre non activée, plafond journalier, clé à corriger) avec la bonne durée de pause. |
| **Quotas suivis** | Budget de jetons lu dans les en-têtes, limites apprises, aucun appel gaspillé sur un modèle qui va refuser. |
| **Tool calling** | Décrivez une fonction Python avec `@tool`, le schéma JSON est déduit. Les erreurs d'outil sont renvoyées au modèle au lieu de planter la conversation. |
| **Lecture par lot en parallèle** | `run_parallel` donne à chaque provider ses travailleurs, avec un plafond de simultanés et un écart entre départs adaptés à ses limites. |
| **Recherche Web gratuite** | Exa, Tavily, LangSearch derrière une interface, quota suivi sur disque, arrêt à 90 % du plafond gratuit. Un outil prêt pour vos modèles. |
| **JSON imparfait** | `read_json` récupère une réponse écrite deux fois ou avec un `null` parasite, sans jamais deviner une réponse coupée. |
| **Données personnelles** | `find_personal_data` repère e-mail, téléphone, IBAN, carte, sécurité sociale (avec leurs clés de contrôle) avant l'envoi. |
| **Tableau de bord** | Quota, latence, pauses, journal et console de test, en local, sans dépendance côté navigateur. |
| **Configurable sans code** | `providers.json` : changez un modèle, retirez ou ajoutez un provider (tout endpoint compatible OpenAI), changez l'ordre. |

## Démarrage

```bash
pip install "ricochet-llm[server]"        # sans le tableau de bord : pip install ricochet-llm
```

Ajoutez au moins une [clé gratuite](#les-providers) dans un fichier `.env` :

```bash
GROQ_API_KEY=...
NVIDIANIM_API_KEY=...        # une clé suffit, plusieurs = plus de marge
```

```bash
ricochet-llm serve --open                 # tableau de bord sur http://127.0.0.1:8765
ricochet-llm chat "Bonjour !"             # une question, une réponse (+ le provider qui a répondu)
```

| Commande | Effet |
|---|---|
| `ricochet-llm serve [--port N] [--open]` | serveur web et tableau de bord (commande par défaut) |
| `ricochet-llm status` | état des providers dans le terminal |
| `ricochet-llm chat "..."` | une question, une réponse |
| `ricochet-llm models nvidia` | **quels modèles listés répondent vraiment** (la liste d'un provider gratuit ne le dit pas) |
| `ricochet-llm search "..."` | recherche Web par API gratuite |

Le tableau de bord écoute uniquement sur votre machine. Il n'a pas d'authentification et ses boutons consomment des quotas : l'écoute sur une autre adresse est refusée sans `--allow-remote`.

Depuis les sources : `git clone https://github.com/Maxichance/ricochet-llm.git && cd ricochet-llm && pip install -e ".[server]"`, puis `cp .env.example .env`.

## Dans votre code

### Une conversation avec des outils

```python
from ricochet_llm import chat_detailed, tool

@tool
def meteo(ville: str) -> dict:
    """Donne la météo d'une ville.

    Args:
        ville: Nom de la ville
    """
    return {"ville": ville, "ciel": "dégagé", "temperature": 18}

history = []                                       # une liste par conversation
r = chat_detailed(history, "Quel temps fait-il à Nantes ?")
print(r.answer)
print(r.provider, r.model, r.trail)                # qui a répondu, et les bascules éventuelles
```

### Un appel isolé, puis un lot en parallèle

```python
from ricochet_llm import generate, run_parallel

def lire(document, provider):                      # exécutée dans un thread
    r = generate(prompt(document), max_tokens=8192, timeout=60, prefer=provider)
    sauvegarder(document, r.text, r.provider, r.model)     # au fil de l'eau : une interruption ne perd rien

bilan = run_parallel(documents, lire, limits={"nvidia": 10})
# {"processed": 812, "errors": 3, "remaining": 0, "per_provider": {...}}
```

`generate` ne renvoie jamais une réponse vide, accepte un délai plus long que celui du chat et bascule entre providers et entre modèles. `run_parallel` s'arrête proprement quand plus aucun provider n'est disponible et vous rend ce qui reste.

### Chercher sur le Web, lire un JSON imparfait, protéger les données

```python
from ricochet_llm import (generate, web_search, register_search_tool,
                          read_json, find_personal_data)

register_search_tool()                             # l'outil `search_web` devient appelable par vos modèles
pages = web_search("routeur llm gratuit", n=5, content=True).results

data = read_json(generate(prompt, max_tokens=8192).text)    # dict, ou None si la réponse est coupée
if find_personal_data(message):                    # e-mail, téléphone, IBAN, carte, sécurité sociale
    ...                                            # ne l'envoyez pas à un free tier
```

Le contrôle du **contenu** d'une réponse reste à vous : `read_json` ne répare que la forme.

## Les providers

| Provider | Où obtenir une clé gratuite | Variable |
|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com) (clé d'essai) | `NVIDIANIM_API_KEY` |
| Ollama Cloud | [ollama.com/settings/keys](https://ollama.com/settings/keys) | `OLLAMACLOUD_API_KEY` |
| Cloudflare Workers AI | [dash.cloudflare.com](https://dash.cloudflare.com/profile/api-tokens) (jeton **et** identifiant de compte) | `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` |
| Gonka | [gonkabroker.com](https://gonkabroker.com) | `GONKA_API_KEY` |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` |
| Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) (vérifiez que l'offre gratuite est active) | `MISTRAL_API_KEY` |
| *Recherche Web* : Exa | [dashboard.exa.ai](https://dashboard.exa.ai) | `EXA_API_KEY` |
| *Recherche Web* : Tavily | [app.tavily.com](https://app.tavily.com) | `TAVILY_API_KEY` |
| *Recherche Web* : LangSearch | [langsearch.com](https://langsearch.com) | `LANGSEARCH_API_KEY` |

Un provider sans clé est simplement ignoré. `NVIDIANIM_API_KEY_EXPIRES=AAAA-MM-JJ` affiche l'échéance d'une clé d'essai dans le tableau de bord, avec une alerte à J-30.

<details>
<summary><b>Voir le détail d'un provider dans le tableau de bord</b></summary>
<br>
<img src="https://raw.githubusercontent.com/Maxichance/ricochet-llm/main/docs/img/providers.png" alt="Lignes dépliées : modèles, jauges de quota, pause avec dernière erreur et bouton de reprise" width="820">
</details>

> **Les offres gratuites changent vite** (modèles retirés, limites modifiées). Testez un modèle par un vrai appel avant de le configurer : `ricochet-llm models <provider>`.
>
> **Confidentialité.** Plusieurs offres gratuites journalisent ou utilisent vos prompts pour entraîner leurs modèles. N'y envoyez pas de données personnelles ou confidentielles sans avoir vérifié leurs conditions. La licence de l'offre gratuite de NVIDIA NIM couvre la recherche, le développement et le test.

## Configuration

Copiez `providers.example.json` en `providers.json` (ou pointez dessus avec `RICOCHET_PROVIDERS`) pour modifier un modèle, retirer un provider, en ajouter un ou changer l'ordre :

```json
{
  "providers": [
    { "name": "groq", "model_heavy": "openai/gpt-oss-120b" },
    { "name": "cerebras", "enabled": false },
    { "name": "monprovider", "env_key": "MONPROVIDER_API_KEY",
      "base_url": "https://api.exemple.com/v1", "model_heavy": "gros-modele" }
  ],
  "order": ["monprovider"]
}
```

Le fichier est validé : une faute de frappe donne une erreur explicite. Sans fichier, `RICOCHET_ORDER=groq,gemini` change la tête de liste. Tous les champs (modèles de repli, options par modèle, limites de jetons, plafond de simultanés...) sont documentés en tête de [`ricochet_llm/catalog.py`](https://github.com/Maxichance/ricochet-llm/blob/main/ricochet_llm/catalog.py).

Vos outils et votre prompt système peuvent vivre dans un plugin, chargé par le serveur et la ligne de commande : `RICOCHET_PLUGIN=examples/demo_tools.py`. Le plugin est du code exécuté tel quel : ne chargez que du code de confiance.

## Documentation

- [Guide technique](https://github.com/Maxichance/ricochet-llm/blob/main/docs/GUIDE.md) : le routage en détail, comment chaque erreur est traitée, comportements des providers, bonnes pratiques de la lecture par lot.
- [Recherche Web](https://github.com/Maxichance/ricochet-llm/blob/main/docs/RECHERCHE_WEB.md) : providers retenus et écartés, quotas, ajouter un moteur.
- [Exemples](https://github.com/Maxichance/ricochet-llm/tree/main/examples) : un plugin d'outils, un usage en bibliothèque.
- [Journal des versions](https://github.com/Maxichance/ricochet-llm/blob/main/CHANGELOG.md).

## Développement

```bash
pip install -e ".[dev]" ruff
ruff check .
python -m unittest discover -s tests -v          # aucun réseau, aucun quota consommé
```

Les tests utilisent de faux providers : ils ne consomment rien.

## Sécurité et licence

Un signalement de faille se fait en privé : voir [SECURITY.md](https://github.com/Maxichance/ricochet-llm/blob/main/SECURITY.md). Code sous licence [MIT](https://github.com/Maxichance/ricochet-llm/blob/main/LICENSE).
