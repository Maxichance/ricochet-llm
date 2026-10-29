# Free web search for the router's agents

The router's models can't search the web: they only call the tools the application gives them.
`ricochet_llm/search.py` provides that tool with the same logic as `router.py`: several providers behind a single
interface, failover on refusal, status readable without a network call.

What a search returns is **pages**, never an AI answer. An agent that uses it produces a simulation:
it does not measure what a consumer assistant answers.

```python
from ricochet_llm import web_search, register_search_tool       # web_search = ricochet_llm.search.search

r = web_search("comparison of free LLM routers", n=5, content=True)
r.status                                                    # SUCCESS | EMPTY | BLOCKED | RATE_LIMITED | ERROR (never raises)
[(p["title"], p["url"], p["snippet"], p.get("content")) for p in r.results]

register_search_tool()      # the `search_web(query)` tool becomes callable by chat() models
```

From the command line: `python -m ricochet_llm search "query" [-n 6] [--content]`. `python -m ricochet_llm status` also shows
the consumption of each configured engine, and the server exposes it at `GET /api/search/status` (no key in the response).

## Providers kept, set aside, and why

Based on each provider's official documentation (pages read; whatever isn't there is marked
"not documented", never assumed). These offers change: check again before relying on them.

| Provider | Free tier (verified) | Card | Renews | Rate limit | Decision |
|---|---|---|---|---|---|
| **Exa** | $10 credit, **reset to $10 on the 1st of each month** (up to 2,500 "lite" searches; "auto" is $7 per 1,000) | "No payment method required" | Yes | 10 requests per second | **Kept**, first |
| **Tavily** | 1,000 credits per month, 1 credit per basic search | "No credit card required" | Yes (1st of the month) | Not documented | **Kept**, second |
| **LangSearch** | Free plan at $0, token allowance **per day** (reset at 00:00 UTC) | Not mentioned | Yes (every day) | 5 requests per second; token cap visible only in the dashboard | **Kept**, third |
| Jina | 10 million tokens with each key, at least 10,000 tokens per search (about 1,000 searches in total) | Not mentioned | No (credit runs out) | 100 per minute | Set aside: not sustainable; terms forbid "automated extraction" of the service |
| Linkup | "4,000 free requests" | Not documented | Not documented | Not documented | Set aside: restrictive terms (no use as a "wrapper", customer data usable for their training) |
| Serper | "2,500 free requests", no card | No | Not specified | Not documented | Set aside: it resells Google results obtained by scraping, which we rule out |
| Brave | "up to $20 of free calls per month" or $5 depending on the page, prepaid credits | Not specified | Yes | 50 per second (paid plan) | Set aside: card probably required, storage rights to be purchased |
| Google Custom Search | - | - | - | - | Set aside: closed to new customers, shutting down on January 1, 2027 |

**To read before scaling up**: the terms of use (storage, reuse of results, use for training) of Exa, Tavily and LangSearch
are not on the documentation pages that were read, and LangSearch has no terms page in its index.

Behavior when exhausted: Exa answers `402` (no billing without a card). Tavily: not documented on the free plan
(its `432` error signals a refusal). LangSearch: `429` until the reset at 00:00 UTC.

## How the router uses it

1. **Order**: `exa, tavily, langsearch` (`RICOCHET_SEARCH_ORDER`). A provider without a key is ignored.
2. **Quota tracked before it is reached**: consumption (dollars or credits, month or day) is written to
   `~/.ricochet/search_usage.json` (`RICOCHET_SEARCH_STATE`). The file lives in the user's home folder, not in the
   project: the quota belongs to the **key**, which several projects may share. The router stops at **90%** of the free
   quota instead of depending on what the provider does at the cap. The real cost is read from the response when it gives it
   (`costDollars` for Exa, `usage.credits` for Tavily, token `usage` for LangSearch).
3. **Failover on refusal**: `401/403` (key to fix, 15-minute pause), `402/432` (quota exhausted: counter pushed to the
   cap, set aside until renewal), `429` (`Retry-After` or 60 s), network error or `5xx` (30 s).
4. **Backup**: when no API is usable, the caller can provide a backup engine (`fallback=`), a function
   `(query, n)` that returns the same `SearchResult`. Original usage example: DuckDuckGo, the only engine whose `robots.txt`
   allows automation, with spacing and a circuit breaker (it raises an anti-bot challenge from the 2nd closely spaced search:
   **never work around it**).
5. **No network call to know the status**: `status()` reads the keys, counters and pauses.
6. What the model sees contains no technical detail: a search that can't be done is returned to it as
   `{"error": "search unavailable", "status": "..."}`.
7. In queries, **give today's date**: without it, models search for the year of their training.

Settings: `RICOCHET_SEARCH_COUNTRY` (Tavily, e.g. `france`, absent by default), `RICOCHET_EXA_TYPE` (`auto` by default).

## Adding a provider

1. Check its **official documentation**: a free tier that renews, no card, limits, reuse terms. Add the result (and
   what isn't documented) to the table above.
2. In `search.py`: an entry in `PROVIDERS` (environment key, URL, period, unit, quota, default cost), a function
   that builds the request, a function that reads the response and returns `(pages, cost)`, then register it in `ADAPTERS`. The
   response key names come from the official schema, not from an example.
3. A test with the response from the official schema (`tests/test_search.py`), with no network call.
4. A real trial with a key before adding it to `DEFAULT_ORDER`.

## Status

| Feature | Status |
|---|---|
| Search router, three adapters, quotas, failover, backup | Implemented and tested (simulated APIs based on the official schemas) |
| Real calls to Exa, Tavily, LangSearch | Adapters written from the official schemas. **A real trial with your keys is still to do** before relying on them |
| Reading the terms of use of the three providers | To do before scaling up |
