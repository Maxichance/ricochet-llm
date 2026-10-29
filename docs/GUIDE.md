# Routeur LLM multi-providers « free tier » — guide technique

> Pour qui contribue au projet ou l'intègre. Démarrage et usage : voir [README](../README.md).

## 1. Principe

Une boucle agentique (chat + tool calling) qui ne dépend d'aucun fournisseur LLM unique. Elle s'appuie sur l'API OpenAI-compatible exposée par plusieurs providers gratuits (un seul SDK : `openai`, pointé sur chaque `base_url`) et bascule de l'un à l'autre.

1. `catalog.py` liste les providers par ordre de préférence (surchargeable par `providers.json`), chacun avec un modèle **heavy**, un modèle **light** et, si besoin, des **modèles de repli** (`extra_models`) du même provider.
2. Avant chaque appel, `_choose_provider_and_model()` sélectionne :
   - **HEAVY** : le premier provider disponible (hors cooldown) dont le quota est inconnu ou ≥ `quota_min`, en privilégiant ceux qui gèrent les tools ; dans ce provider, le **premier modèle utilisable** (heavy, puis `extra_models`) ;
   - **LIGHT** (tous sous le seuil) : celui qui a le plus de quota restant, sur son modèle léger.
   Un modèle écarté ne met **pas** tout le provider de côté : l'état (pause, limite de jetons apprise) est tenu **par modèle**. Le provider n'est en pause que lorsque plus aucun de ses modèles ne peut répondre, jusqu'au premier retour.
3. Chaque erreur d'appel est **classée** (`_classify_error`) ; la conversation n'est jamais perdue :

   | Erreur | Réaction |
   |---|---|
   | `429` | le **modèle** est écarté (`Retry-After`, 60 s par défaut) ; un autre modèle du provider est essayé tout de suite ; sans autre modèle utilisable, le provider est en pause jusqu'au plus proche retour d'un modèle, quota à 0 |
   | `413` « Limit N » | la limite N est **apprise** pour ce modèle, qui est écarté 30 min (les demandes plus grandes ne lui sont plus envoyées) ; mêmes règles que le 429 |
   | réponse **vide** | modèle écarté 5 min, modèle suivant (un modèle qui raisonne a dépensé tous ses jetons avant d'écrire) |
   | réseau / 5xx / timeout | 1 nouvelle tentative avec backoff, puis cooldown 30 s |
   | `404` | modèle retiré du catalogue : écarté 15 min s'il y en a d'autres, sinon cooldown du provider 15 min |
   | `401` / `403` | cooldown du provider 15 min (clé à corriger) |
   | `400` / `410` / `422` | bascule pour cet appel seulement, sans cooldown |
   | autre exception (bug local) | remontée telle quelle |

   Un provider écarté pendant un appel n'est jamais re-tenté dans ce même appel : la boucle se termine toujours (sinon `LLMRouterError` avec la dernière erreur de chaque provider). En cas d'échec, l'historique est remis dans son état d'origine.
4. Les tool calls sont exécutés localement (registre `tools.py`) et réinjectés jusqu'à `finish_reason == "stop"` (8 tours max). Un outil inconnu, des arguments invalides ou une exception d'outil sont renvoyés **au modèle** comme résultat. Pendant un échange d'outils, on ne bascule jamais vers un provider sans tool calling.

## 2. Architecture

- `catalog.py` (données) → `router.py` (moteur et état) → `server.py` / `__main__.py` (entrées) → `web/` (interface). `parallel.py` répartit des traitements par lot entre les providers (section 6).
- Trois portes d'entrée dans le moteur : `chat` (conversation avec outils), `generate` (un appel isolé : extraction, classification, lecture de documents) et `probe_models` (diagnostic : quels modèles listés répondent vraiment).
- État observable par provider, tenu dans `router.py` sous un verrou : quota restant, cooldown, dernière erreur, derniers en-têtes `x-ratelimit-*`, sonde de quota, statistiques (ok, erreurs, tokens, latence), fenêtre d'appels, plus un journal d'événements. `quota_status()` le restitue, enrichi (jauges, expiration de clé) ; l'interface l'interroge via `GET /api/status` toutes les 3 s.
- **Jauges de quota** : chaque paire `x-ratelimit-remaining-X` / `x-ratelimit-limit-X` devient une jauge (`quota_labels` donne les libellés). OpenRouter ne renvoie rien en en-tête : une sonde (`probe`) lit `GET /key` (`free_model_daily_requests`, `expires_at`), au plus une fois par minute.
- **Expiration de clé** : `<NOM_DE_LA_CLÉ>_EXPIRES=AAAA-MM-JJ` dans `.env`, ou `expires_at` fourni par la sonde.
- **Sécurité** : le serveur écoute en `127.0.0.1` par défaut et n'a pas d'authentification (ses actions consomment des quotas). Ne l'exposez pas sans ajouter une authentification.

## 3. Ajouter ou modifier un provider

Sans toucher au code : `providers.json` (voir `providers.example.json` et la docstring de `catalog.py`) — surcharge un provider, en retire un, en ajoute un, change l'ordre. Le fichier est validé au chargement (`ConfigError` explicite). Pour un provider livré avec le paquet, ajouter une entrée à `BUILTIN_SPECS`. **Avant de choisir un modèle, le tester par un vrai appel** (voir pièges). Sans en-tête de quota, laisser `quota_header` à `null` : seuls les 429 déclencheront alors le cooldown. Lancer `python -m ricochet models <provider>` : **la liste d'un provider gratuit ne dit pas ce qui répond**. Pour un modèle qui raisonne, renseigner `model_options` (couper le raisonnement) ; pour un provider à budget par modèle, `token_limits` ; pour un provider à débit limité, `max_concurrency` et `min_interval_s`. Pour une sonde dédiée, ajouter une fonction dans `_PROBES` (`router.py`).

## 4. Intégrer dans un autre projet

`pip install -e .` puis `from ricochet import chat, tool, set_system_prompt`. Déclarez vos outils avec `@tool` (schéma déduit de la signature et de la docstring). Le client `openai` est synchrone : dans une application asynchrone, appelez `chat` via `run_in_threadpool` / `run_in_executor` (comme `server.py`). Les sessions du serveur sont en mémoire (suffisant pour un outil interne mono-instance ; sur plusieurs instances, prévoir Redis/Firestore). Ne pas envoyer deux messages en parallèle sur la même session (le serveur renvoie 409). Sans outil enregistré, aucun paramètre `tools` n'est envoyé aux providers.

## 5. Comportements des providers à connaître

Les offres gratuites changent vite (modèles retirés, limites modifiées) : **testez un modèle par un vrai appel avant de le configurer** (`python -m ricochet models <provider>`), la liste `/models` d'un provider n'est pas fiable.

**Refus et pauses**
- **Quel refus dit quoi** : un **402** ou une limite par minute à zéro (`x-ratelimit-limit-req-minute = 0`) = offre non activée, pause d'une heure. Un **429 sur un plafond journalier** (Gemini `GenerateRequestsPerDayPerProjectPerModel`, Cloudflare « daily free allocation », OpenRouter `free-models-per-day`) ne revient pas avant des heures : le retenter chaque minute ne ferait que perdre un appel. Un **429 sans `Retry-After`** mais avec `x-ratelimit-reset-tokens` (« 46.5s », « 1m30s ») : on attend la remise à zéro annoncée.
- **Le budget de jetons annoncé** (`x-ratelimit-remaining-tokens` / `-reset-tokens`) est retenu par modèle après chaque réponse réussie : un modèle qui n'a plus de quoi traiter la demande n'est pas appelé. `x-ratelimit-limit-tokens` fixe la plus grande demande acceptable, sans attendre un 413.
- **Clés d'essai à durée limitée** (NVIDIA, OpenRouter) : passée la date, 401/403 → pause de 15 min en boucle, sans réparation automatique. `<NOM_DE_LA_CLÉ>_EXPIRES=AAAA-MM-JJ` affiche l'échéance.
- **Quotas par fenêtre différente** (Groq : jetons/min et requêtes/jour ; Cerebras : jetons/jour) : les `quota_min` du catalogue sont calibrés sur ces fenêtres, à revoir si on change de modèle. Un quota lu depuis plus de 5 min redevient « inconnu » et le provider est re-sondé.
- **Le SDK ne doit pas re-tenter lui-même** : clients créés avec `max_retries=0`, sinon il dormirait sur un 429 avant que le routeur bascule.

**Réponses des modèles**
- **Modèles qui raisonnent** (gpt-oss, Nemotron…) : `max_tokens` inclut les jetons de raisonnement ; une réponse vide déclenche une bascule. Pour les Nemotron de NVIDIA, `model_options` coupe le raisonnement (`enable_thinking: false`) ; un délai plus long n'y change rien.
- **`<think>...</think>` dans `content`** : jamais rendu comme réponse ; une réponse faite de la seule réflexion compte comme vide (bascule).
- **Petits modèles** : ils renvoient parfois le JSON d'un tool call en texte brut. `cleaning.strip_tool_artifacts` retire **uniquement** les JSON reconnus comme tool calls (nom connu, ou `{"type": "function", …}`) et les balises `<function=…>` / `<tool_call>` ; le code et le JSON légitimes sont conservés. Si le nettoyage vide la réponse, on bascule et on relance une fois avec « Réponds à ma question précédente ».
- **Gemini et les tools** : Gemini 3.x exige un champ `thought_signature` sur le second tour d'un échange d'outils, que le routeur ne rejoue pas : `supports_tools: False`. Chaîne de repli `gemini-3.6-flash`, `3.5-flash`, `3.1-pro-preview` (le plus capable, mais son quota gratuit par jour est très court).
- **Mistral** : une clé valide peut n'avoir aucun droit de chat tant que le plan gratuit n'est pas activé ; le routeur met alors Mistral en pause une heure et utilise les autres.
- **OpenRouter** : un 429 « temporarily rate-limited upstream » vient souvent du fournisseur amont du modèle gratuit, pas de votre quota.
- **Cerebras** : plus d'offre gratuite permanente d'après les sources récentes : carte « désactivée » sans clé.

**Ordre par défaut** : Groq, NVIDIA, Ollama, Cloudflare, OpenRouter, Gonka (6 requêtes/min et 1 M de jetons par mois : dernier des gratuits), Gemini, Mistral. `RICOCHET_ORDER=groq,gemini` change la tête de liste sans fichier.

**Confidentialité** : plusieurs offres gratuites journalisent ou utilisent les prompts pour entraîner leurs modèles. N'y envoyez pas de données personnelles sans avoir vérifié leurs conditions (`ricochet.privacy` repère les plus évidentes).

## 5 bis. Bonnes pratiques de la lecture par lot

- **Un JSON valide mais vide n'est pas « aucun résultat ».** Un modèle trop faible rend du JSON bien formé sans rien extraire. Le routeur ne peut pas le voir (la réponse n'est pas vide) : **validez le contenu côté appelant** (une citation retrouvée dans la source, par exemple) et comparez un modèle à une référence sur quelques entrées AVANT de l'activer.
- **Les petits modèles lisent moins.** Bons en dernier repli, pas en premier choix.
- **Une limite de débit n'est pas une limite de simultanéité.** « 40 requêtes par minute » porte sur le rythme de DÉPART : réglez `max_concurrency` et `min_interval_s` pour rester sous la limite plutôt que de vous borner à 1 ou 2 travailleurs. Une salve passe, un flux soutenu atteint la limite.
- **Un budget de jetons par minute PAR MODÈLE** (Groq) : un 429 ou un 413 sur un modèle ne dit rien des autres, d'où l'état par modèle.
- **Le délai du chat n'est pas celui d'une lecture.** Le délai du catalogue est calibré pour le chat (on bascule vite), pas pour une longue réponse JSON. `generate(timeout=60)` allonge le délai pour cet appel, ne le raccourcit jamais.
- **Comptez par le provider qui a RÉPONDU**, pas par celui qu'on avait préféré : lisez `GenerateResult.provider` et `.model`. Les durées mesurées autour d'un appel incluent les bascules.
- **Comparez des modèles sur les mêmes entrées** : l'écart entre échantillons différents dépasse souvent celui entre modèles.
- **Quotas et licences** : NVIDIA ne renvoie aucun en-tête de quota et la licence de son offre gratuite est « recherche, développement et test » ; Gonka plafonne les requêtes par minute et les jetons par mois ; Cloudflare plafonne par jour. Ne laissez pas un traitement par lot épuiser ce dont le chat a besoin en repli : lancez-le la nuit, ou excluez ces providers (`only=`).
- **Les clés collées dans un `.env`** : une valeur beaucoup trop longue (jeton collé plusieurs fois, commentaire, guillemets) donne 401 ou 400 « Invalid format for Authorization header ». Vérifiez la longueur et les caractères sans afficher la clé. Une variable mal nommée est acceptée via `env_aliases`.
- **Windows : un clic dans la console peut tout figer.** Le mode « édition rapide » met la console en sélection : le premier `print` reste bloqué tant qu'on n'appuie pas sur Entrée, et un programme qui écrit dans la console, même dans un thread, peut figer tous ses appels en cours. Désactivez l'édition rapide, ou utilisez Windows Terminal.
- **Enregistrez au fil de l'eau** : écrivez le résultat de chaque élément dès qu'il est lu (écriture atomique : fichier temporaire puis renommage). Une interruption ne perd alors que les requêtes en cours.
- **Tests : `load_dotenv()` modifie `os.environ` pour toute la suite.** Un test qui importe un module qui charge le `.env` réel injecte de vraies clés dans les tests suivants. Neutralisez-le (`mock.patch("dotenv.load_dotenv")`) avant l'import.

## 6. Lectures par lot : `generate` et `run_parallel`

```python
from ricochet import generate, run_parallel

def lire(document, provider):                      # exécutée dans un thread
    r = generate(prompt(document), system=CONSIGNE, max_tokens=8192, timeout=60, prefer=provider)
    if r.truncated:                                # arrêtée sur max_tokens : le JSON peut être coupé
        raise ValueError("réponse tronquée")
    sauvegarder(document, r)                       # au fil de l'eau ; noter r.provider et r.model

bilan = run_parallel(documents, lire, limits={"nvidia": 10})
# {"processed": …, "errors": …, "remaining": …, "interrupted": False, "per_provider": {"groq": …, …}}
```

- `generate` bascule entre providers **et** entre modèles comme `chat`, sans outils. Il ne renvoie **jamais** une réponse vide (le modèle est écarté, le suivant essayé). `timeout` allonge le délai d'un appel ; `only` restreint les providers ; `prefer` en désigne un à essayer d'abord.
- `run_parallel` donne à chaque provider **ses** travailleurs (`max_concurrency` du catalogue, ou `limits`), tous tirant dans la même file (l'ordre est conservé), avec un écart minimal entre départs (`min_interval_s`). Un provider en pause ne prend pas de travail ; plus aucun provider disponible, ou `Interrupted` levée par `work` : arrêt propre, les éléments restants sont rendus (`remaining`). Une erreur sur un élément est comptée sans arrêter les autres.
- **Choisissez les providers du lot** (`only=`) : un provider à quota mensuel court (Gonka) ou plafonné par jour (Cloudflare) est un repli du chat, pas un moteur de rattrapage.
- Avant de lancer, une tranche de ~100 éléments, puis un contrôle de qualité **par provider et par modèle** (éléments par document, part de citations retrouvées, réponses illisibles).

## 6 bis. Réponses JSON imparfaites : `read_json`

Certains modèles écrivent leur réponse deux fois (l'objet complet, un guillemet, une seconde copie), la faisait suivre d'un champ en trop (`...}}"produits":[]}`) ou glissait un `null` entre deux paires clé/valeur. `read_json(texte)` lit strictement d'abord (inchangé pour une réponse valide), puis seulement en cas d'échec garde le premier objet complet. Une réponse coupée (`max_tokens`) ou qui boucle reste `None` : on la relit. Il ne dit rien du **contenu** : un JSON valide peut contenir des valeurs inventées, le contrôle de fond (la citation est-elle dans la source ?) reste à vous.

## 7. Checklist

1. `pip install -r requirements.txt`, créer `.env` depuis `.env.example`, renseigner au moins une clé.
2. Lancer `python -m ricochet serve --open`, cliquer **Tester** sur chaque carte : clé valide et latence correcte ?
3. Déclarer vos outils (`@tool`) dans un plugin (`RICOCHET_PLUGIN`) ou dans votre code.
4. `python -m unittest discover -s tests -v` après toute modification du moteur.
5. Avant tout traitement par lot : `python -m ricochet models <provider>` (modèles réellement appelables), une tranche de ~100 éléments, un bilan par provider ET par modèle.
6. Avant d'envoyer un texte saisi par un utilisateur à un free tier : `find_personal_data(texte)`. Pour la recherche Web : `python -m ricochet search "test"` avec une vraie clé (les adaptateurs n'ont été testés que contre des réponses simulées).
7. Tester la bascule en réel : mettre un `quota_min` très haut sur le premier provider (`catalog.py`) et vérifier dans le journal que le suivant prend le relais.
