# Contributing

The project is collaborative: anyone can take it further, including in depth. New providers, new models, other API formats, other routing strategies, fixes, tests, documentation: propose it and we'll talk.

## Proposing a change

1. Fork the repository, then create a branch.
2. Install the project: `pip install -e ".[dev]" ruff`.
3. Before opening the pull request:
   ```
   ruff check .
   python -m unittest discover -s tests -v
   ```
4. Open a pull request against `main`.

For a large change, an issue beforehand lets you discuss it before writing the code, but it isn't required.

## Providers and models

- **For your own use**: a `providers.json` file is enough, with no code change (see `providers.example.json` and [docs/GUIDE.md](docs/GUIDE.md), section 3).
- **To share it**: add it to `ricochet_llm/catalog.py`, or change the existing model chains.
- **An API that isn't OpenAI-compatible** is welcome too: the router speaks that format today, and an adapter for another format is a good contribution.

A tip: test with a real call before proposing (`python -m ricochet_llm models <provider>`), as a provider's `/models` list isn't always reliable. Also check that the free tier's terms of use allow this usage.

## Ground rules

- Everything committed is public: **no keys, no `.env`, no personal data**.
- CI must pass (Python 3.9 and 3.12).
- Report security vulnerabilities privately, not in a public issue: see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your code is published under the project's [MIT](LICENSE) license.
