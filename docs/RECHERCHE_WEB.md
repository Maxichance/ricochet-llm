# Recherche Web par API gratuite pour les agents du routeur

Les modèles du routeur ne savent pas chercher sur le Web : ils n'appellent que les outils que l'application leur donne.
`ricochet/search.py` fournit cet outil avec la même logique que `router.py` : plusieurs providers derrière une seule
interface, bascule sur un refus, état consultable sans appel réseau.

Ce que rend une recherche, ce sont des **pages**, jamais une réponse d'IA. Un agent qui s'en sert produit une simulation :
elle ne mesure pas ce que répond un assistant grand public.

```python
from ricochet import web_search, register_search_tool       # web_search = ricochet.search.search

r = web_search("comparatif des routeurs LLM gratuits", n=5, content=True)
r.status                                                    # SUCCESS | EMPTY | BLOCKED | RATE_LIMITED | ERROR (ne lève jamais)
[(p["title"], p["url"], p["snippet"], p.get("content")) for p in r.results]

register_search_tool()      # l'outil `search_web(query)` devient appelable par les modèles de chat()
```

En ligne de commande : `python -m ricochet search "requête" [-n 6] [--content]`. `python -m ricochet status` affiche aussi la
consommation de chaque moteur configuré, et le serveur la donne à `GET /api/search/status` (aucune clé dans la réponse).

## Providers retenus, écartés, et pourquoi

D'après la documentation officielle de chaque fournisseur (pages lues ; ce qui n'y figure pas est noté
« non documenté », jamais supposé). Ces offres changent : à revérifier avant de s'y fier.

| Fournisseur | Offre gratuite (vérifiée) | Carte | Se renouvelle | Limite de débit | Décision |
|---|---|---|---|---|---|
| **Exa** | 10 $ de crédit, **remis à 10 $ le 1er de chaque mois** (jusqu'à 2 500 recherches « lite » ; « auto » 7 $ les 1 000) | « No payment method required » | Oui | 10 requêtes par seconde | **Retenu**, premier |
| **Tavily** | 1 000 crédits par mois, 1 crédit par recherche simple | « No credit card required » | Oui (1er du mois) | Non documentée | **Retenu**, deuxième |
| **LangSearch** | Plan gratuit à 0 $, allocation de jetons **par jour** (remise à zéro à 00:00 UTC) | Non mentionnée | Oui (chaque jour) | 5 requêtes par seconde ; plafond de jetons visible seulement dans le tableau de bord | **Retenu**, troisième |
| Jina | 10 millions de jetons offerts avec chaque clé, 10 000 jetons au minimum par recherche (soit environ 1 000 recherches au total) | Non mentionnée | Non (crédit qui s'épuise) | 100 par minute | Écarté : pas durable ; conditions qui interdisent « l'extraction automatisée » du service |
| Linkup | « 4 000 requêtes gratuites » | Non documenté | Non documenté | Non documentée | Écarté : conditions restrictives (pas d'usage comme « wrapper », données client utilisables pour leur entraînement) |
| Serper | « 2 500 requêtes gratuites », sans carte | Non | Non précisé | Non documentée | Écarté : il revend des résultats Google obtenus par extraction, ce que nous excluons |
| Brave | « jusqu'à 20 $ d'appels gratuits par mois » ou 5 $ selon la page, crédits prépayés | Non précisé | Oui | 50 par seconde (plan payant) | Écarté : carte probablement exigée, droits de stockage à acheter |
| Google Custom Search | - | - | - | - | Écarté : fermé aux nouveaux clients, arrêt le 1er janvier 2027 |

**À lire avant une montée en volume** : les conditions d'utilisation (stockage, réutilisation des résultats,
usage pour l'entraînement) d'Exa, de Tavily et de LangSearch ne figurent pas sur les pages de documentation lues, et LangSearch
n'a aucune page de conditions dans son index.

Comportement à l'épuisement : Exa répond `402` (pas de facturation sans carte). Tavily : non documenté sur le plan gratuit
(son erreur `432` indique un refus). LangSearch : `429` jusqu'à la remise à zéro de 00:00 UTC.

## Comment le routeur s'en sert

1. **Ordre** : `exa, tavily, langsearch` (`RICOCHET_SEARCH_ORDER`). Un provider sans clé est ignoré.
2. **Quota suivi avant d'être atteint** : les consommations (dollars ou crédits, mois ou jour) sont écrites dans
   `~/.ricochet/search_usage.json` (`RICOCHET_SEARCH_STATE`). Le fichier est dans le dossier de l'utilisateur et non dans le
   projet : le quota appartient à la **clé**, que plusieurs projets peuvent partager. Le routeur s'arrête à **90 %** du quota
   gratuit au lieu de dépendre de ce que fait le provider au plafond. Le coût réel est lu dans la réponse quand elle le donne
   (`costDollars` chez Exa, `usage.credits` chez Tavily, `usage` en jetons chez LangSearch).
3. **Bascule sur un refus** : `401/403` (clé à corriger, pause de 15 minutes), `402/432` (quota épuisé : compteur poussé au
   plafond, écarté jusqu'au renouvellement), `429` (`Retry-After` ou 60 s), erreur réseau ou `5xx` (30 s).
4. **Secours** : quand aucune API n'est utilisable, l'appelant peut fournir un moteur de secours (`fallback=`), une fonction
   `(requête, n)` qui rend le même `SearchResult`. Exemple d'usage d'origine : DuckDuckGo, seul moteur dont le `robots.txt`
   autorise l'automatisation, avec espacement et disjoncteur (il pose un défi anti-robot dès la 2e recherche rapprochée :
   **ne jamais le contourner**).
5. **Aucun appel réseau pour connaître l'état** : `status()` lit les clés, les compteurs et les pauses.
6. Ce que voit le modèle ne contient aucun détail technique : une recherche impossible lui est rendue comme
   `{"error": "recherche indisponible", "status": "..."}`.
7. Dans les requêtes, **donner la date du jour** : sans elle, les modèles cherchent l'année de leur entraînement.

Réglages : `RICOCHET_SEARCH_COUNTRY` (Tavily, ex. `france`, absent par défaut), `RICOCHET_EXA_TYPE` (`auto` par défaut).

## Ajouter un provider

1. Vérifier dans sa **documentation officielle** : offre gratuite qui se renouvelle, absence de carte, limites, conditions de
   réutilisation. Ajouter le résultat (et ce qui n'est pas documenté) au tableau ci-dessus.
2. Dans `search.py` : une entrée de `PROVIDERS` (clé d'environnement, URL, période, unité, quota, coût par défaut), une fonction
   qui construit la requête, une fonction qui lit la réponse et rend `(pages, coût)`, puis l'inscrire dans `ADAPTERS`. Le nom
   des clés de réponse vient du schéma officiel, pas d'un exemple.
3. Un test avec la réponse du schéma officiel (`tests/test_search.py`), sans appel réseau.
4. Un essai réel avec une clé avant de l'ajouter à `DEFAULT_ORDER`.

## État

| Fonction | État |
|---|---|
| Routeur de recherche, trois adaptateurs, quotas, bascule, secours | Implémenté et testé (API simulées d'après les schémas officiels) |
| Appels réels à Exa, Tavily, LangSearch | Adaptateurs écrits d'après les schémas officiels. **Un essai réel avec vos clés reste à faire** avant de compter dessus |
| Lecture des conditions d'utilisation des trois providers | À faire avant une montée en volume |
