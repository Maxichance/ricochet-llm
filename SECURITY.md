# Sécurité

## Signaler une faille

Ouvrez un [signalement privé](https://github.com/Maxichance/ricochet-llm/security/advisories/new) sur GitHub. Merci de ne pas publier la faille dans une issue avant correction.

## Ce qu'il faut savoir en l'utilisant

- **Le tableau de bord n'a pas d'authentification.** Il écoute sur `127.0.0.1` uniquement ; `--allow-remote` est refusé par défaut. Si vous l'exposez, placez-le derrière un proxy authentifié : ses boutons consomment vos quotas.
- **Les clés restent dans `.env`**, qui est ignoré par git. Ne le publiez jamais. Aucune clé n'est renvoyée par l'API (`/api/status`, `/api/search/status`).
- **`RICOCHET_PLUGIN` exécute du code tel quel** : ne chargez que du code de confiance.
- **Vos requêtes partent chez des tiers.** Plusieurs offres gratuites journalisent ou réutilisent les échanges ; voir le README. `ricochet.privacy` repère les données personnelles évidentes avant envoi, sans garantie d'exhaustivité.
