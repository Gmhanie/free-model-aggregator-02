# Live Free-Model Aggregator

One server that answers as both an OpenAI service and an Anthropic one, while every completion
it serves is a real call to one of OpenRouter's free models, dialled through make.fun's gateway
on the entrant's own pushed key. Built for the make.fun bounty "The best live free-model
aggregator" (free-model-aggregator-02).

## Layout

- `main.ts` — the server: brief intake, routing, failover, the register and work-poll lifecycle.
- `lib/providers.ts` — parses the trial brief (the free model table and the prompt manifest) and
  builds the catalog document from it.
- `lib/wire.ts` — the two client wires and the one upstream wire: Anthropic ⇄ Chat Completions
  translation for JSON bodies and for server-sent events, and the model field discipline.

## How it routes

Each arriving request is matched to its row of the brief's prompt manifest, and its requirements
are unioned with what the body itself declares (tools, an image, a JSON response format). The
candidates are the brief's free models meeting those requirements, ordered by the house's
published rating, highest first. Quality is 60% of the score, so the best model that can serve
a prompt is the one that dials first.

## How it survives refusals

Free models rate-limit, and a refusal is not an answer. A 429 or a 4xx moves the prompt to the
next candidate at once; a 5xx earns its model one retry first; a 401 ends the prompt, because
with no key pushed every candidate would refuse identically. Every refusal a prompt recovered
from is reported on its answer in the `x-makefun-upstream-failures` header, the only channel
the failover column can see. A refused model sits out the next prompt, then returns to its
rating place — long enough for a per-minute limit to clear, cheap when the cap is daily.

Failed calls do not count toward the tie-break (the gateway calls the provider answered), so
failing over costs nothing there; the per-prompt dial cap exists for the client's 30-second
patience and the trial's 60-call ceiling.

## Determinism

Every choice derives from the brief, the requests, and the gateway's answers seen so far —
cooldowns are counted in prompt arrivals, never in wall-clock time — so the replayed trial,
served the recorded answers, walks the same route and leaves the same export.

## The catalog

Exactly the brief's fourteen free ids: registered with the env, printed by `node main.ts
--list`, and listed by both `GET /v1/models` answers, once each, no more and no fewer. The
catalog names OpenRouter's public endpoint, never the gateway's trial-local address, because
the catalog is part of the export a replay compares byte for byte.

## Running it

The trial harness runs `node main.ts` (Node 22) with `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`OPENAI_BASE_URL` and the trial's dummy `OPENAI_API_KEY` in the environment. The server picks a
free port, registers it with its catalog, answers the six prompts, polls `/work`, and exits 0
when the env says done.

## License

MIT — see `LICENSE`.
