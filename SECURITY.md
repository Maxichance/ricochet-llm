# Security

## Reporting a vulnerability

Open a [private report](https://github.com/Maxichance/ricochet-llm/security/advisories/new) on GitHub. Please don't disclose the vulnerability in a public issue before it is fixed.

## What to know when using it

- **The dashboard has no authentication.** It only listens on `127.0.0.1`; `--allow-remote` is refused by default. If you expose it, put it behind an authenticated proxy: its buttons consume your quotas.
- **Keys stay in `.env`**, which git ignores. Never publish it. No key is returned by the API (`/api/status`, `/api/search/status`).
- **`RICOCHET_PLUGIN` runs code as is**: only load code you trust.
- **Your requests go to third parties.** Several free tiers log or reuse exchanges; see the README. `ricochet_llm.privacy` spots obvious personal data before sending, with no guarantee of completeness.
