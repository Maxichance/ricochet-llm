# Contribuer

Les contributions sont bienvenues : correctifs, tests, documentation, nouveaux providers.

## Proposer une modification

1. Faites un fork du dépôt, puis créez une branche.
2. Installez le projet : `pip install -e ".[dev]" ruff`.
3. Vérifiez avant d'ouvrir la pull request :
   ```
   ruff check .
   python -m unittest discover -s tests -v
   ```
4. Ouvrez une pull request vers `main`. Rien n'est fusionné sans relecture du mainteneur.

Pour un changement important, ouvrez d'abord une issue.

## Ajouter un provider

- **Pour votre usage** : un fichier `providers.json` suffit, sans modifier le code (voir `providers.example.json` et [docs/GUIDE.md](docs/GUIDE.md), section 3).
- **Pour le proposer au dépôt** : ajoutez-le à `ricochet_llm/catalog.py`, avec un test.

Un provider proposé doit :
- exposer une API compatible OpenAI ;
- avoir une offre gratuite dont les conditions d'utilisation autorisent cet usage ;
- avoir été testé par un vrai appel (`python -m ricochet_llm models <provider>`), la liste `/models` n'étant pas fiable.

## Règles

- Tout ce qui est commité est public : **aucune clé, aucun `.env`, aucune donnée personnelle**.
- Pas de mesures, de captures ou de récits d'essais dans le dépôt : il décrit l'outil.
- Messages de commit courts, à l'impératif ou au présent.
- Ajoutez ou adaptez les tests ; la CI doit passer sur Python 3.9 et 3.12.
- Les failles de sécurité ne se signalent pas dans une issue publique : voir [SECURITY.md](SECURITY.md).

## Licence

En contribuant, vous acceptez que votre code soit publié sous la licence [MIT](LICENSE) du projet.
