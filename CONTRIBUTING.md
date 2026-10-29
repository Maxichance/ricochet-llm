# Contribuer

Le projet est collaboratif : tout le monde peut le faire évoluer, y compris en profondeur. Nouveaux providers, nouveaux modèles, autres formats d'API, autres stratégies de routage, correctifs, tests, documentation : proposez, on en discute.

## Proposer une modification

1. Faites un fork du dépôt, puis créez une branche.
2. Installez le projet : `pip install -e ".[dev]" ruff`.
3. Avant d'ouvrir la pull request :
   ```
   ruff check .
   python -m unittest discover -s tests -v
   ```
4. Ouvrez une pull request vers `main`.

Pour un changement important, une issue au préalable permet d'en parler avant d'écrire le code, mais ce n'est pas obligatoire.

## Providers et modèles

- **Pour votre usage** : un fichier `providers.json` suffit, sans modifier le code (voir `providers.example.json` et [docs/GUIDE.md](docs/GUIDE.md), section 3).
- **Pour le partager** : ajoutez-le à `ricochet_llm/catalog.py`, ou changez les chaînes de modèles existantes.
- **Une API qui n'est pas compatible OpenAI** est la bienvenue aussi : le routeur parle aujourd'hui ce format, un adaptateur pour un autre format est un bon sujet de contribution.

Un conseil : testez par un vrai appel avant de proposer (`python -m ricochet_llm models <provider>`), la liste `/models` d'un provider n'étant pas toujours fiable. Et vérifiez que les conditions d'utilisation de l'offre gratuite permettent cet usage.

## À respecter

- Tout ce qui est commité est public : **aucune clé, aucun `.env`, aucune donnée personnelle**.
- La CI doit passer (Python 3.9 et 3.12).
- Les failles de sécurité ne se signalent pas dans une issue publique : voir [SECURITY.md](SECURITY.md).

## Licence

En contribuant, vous acceptez que votre code soit publié sous la licence [MIT](LICENSE) du projet.
