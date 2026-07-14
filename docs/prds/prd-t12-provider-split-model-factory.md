# PRD T12 — Provider split: supervisor on Gemini Flash via a per-provider model factory

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T11 (clean measurement baseline). Decision record 2026-07-06 §2.

## Problem Statement

As a researcher, nearly every question I ask stalls for 30–60 seconds at a time: the supervisor and both sub-agents share one Groq free-tier token-per-minute bucket, so a single satellite workflow exhausts the minute's budget and every subsequent model call — including the supervisor's own routing call — sits in a rate-limit retry sleep. As the system owner, my README claims the supervisor runs on Gemini while the code runs it on Groq, and I have no way to change providers per agent without editing agent construction code.

## Solution

Split the LLM load across providers along the roles' actual needs: the supervisor (cheap routing and synthesis, high call frequency) moves to Gemini 2.5 Flash, whose free tier has an order of magnitude more headroom; the earthdata agent keeps the whole Groq budget to itself on the large tool-use model; the ground agent stays on the small Groq model. A single model factory turns a settings entry (provider + model name) into a chat model, so provider choice is configuration, not code — free tiers for development, paid models for demos, per the agreed budget posture.

## User Stories

1. As a researcher, I want my question routed and synthesized without waiting behind the satellite agent's token spend, so that simple questions come back in seconds.
2. As a researcher, I want the earthdata agent to have the full Groq rate budget to itself, so that a multi-step satellite workflow completes without mid-workflow rate-limit sleeps.
3. As the system owner, I want each agent's provider and model set in configuration, so that switching providers or upgrading a model for a demo is an environment change, not a code change.
4. As the system owner, I want the supervisor on a provider whose free tier tolerates the supervisor's high call frequency, so that development stays on free tiers per the agreed budget posture.
5. As the developer, I want one factory that maps a provider+model setting to a constructed chat model, so that agent-construction code never mentions a concrete provider class again.
6. As the developer, I want a misconfigured provider name to fail loudly at startup, so that a typo is discovered at boot, not mid-conversation.
7. As the system owner, I want the README's architecture description to match the running configuration, so that documentation stops contradicting the code.
8. As the developer, I want the scripted eval run against the new supervisor configuration before the change merges, so that the provider move is justified by measurement, not vibes.
9. As an operator reading logs, I want the startup log to state each agent's resolved provider and model, so that I can confirm at a glance which bucket each role is spending from.

## Implementation Decisions

- A model factory in configuration/agent-construction space: input is a per-agent setting carrying provider and model identifier; output is a constructed chat model. Supported providers initially: Groq and Google Gemini. Unknown provider names raise a configuration error at startup (matching the existing fail-at-boot posture for the MCP connection).
- Defaults per decision record 2026-07-06: supervisor → Gemini 2.5 Flash; earthdata agent → the large Groq tool-use model; ground agent → the small Groq model. All three remain individually overridable by environment variable, preserving the existing variable names where they exist.
- The Google API key becomes required-at-startup only when a configured agent resolves to the Gemini provider; the existing required-variable validation is extended, not duplicated.
- Supervisor middleware notes that were provider-specific (empty-contents handling) are kept accurate for the provider actually in use.
- README architecture section updated to describe the split truthfully.

## Technical Implementation Guide

Paths current as of `talking-to-air-v2` (c75f4da + working tree).

- **The factory.** New module (suggest `Backend/config/model_factory.py`): `build_chat_model(provider: str, model: str, settings: Settings)` returning a LangChain chat model. `provider == "groq"` → `ChatGroq(model=..., groq_api_key=settings.groq_api_key)` (current construction in `agents/supervisor_agent.py::build_agent`, `agents/earthdata_agent.py`, `agents/ground_sensor_agent.py` — replace all three). `provider == "google"` → `ChatGoogleGenerativeAI(model=..., google_api_key=settings.google_api_key)` from `langchain-google-genai` (add to `Backend/requirements.txt`; it is not there yet — only `langchain-groq` is). Unknown provider → raise `ConfigurationError` (already defined in `Backend/config/settings.py`).
- **Settings.** Add per-agent provider fields alongside the existing model fields in `config/settings.py`: `SUPERVISOR_MODEL_PROVIDER` (default `google`), `EARTHDATA_AGENT_PROVIDER` (default `groq`), `GROUND_AGENT_PROVIDER` (default `groq`). Change the supervisor default model: `LLM_MODEL` default from `openai/gpt-oss-120b` → `gemini-2.5-flash`. Keep the existing env var names for models (`LLM_MODEL`, `EARTHDATA_AGENT_MODEL`/`SATELLITE_AGENT_MODEL` fallback chain, `GROUND_AGENT_MODEL`).
- **Startup validation.** `Settings.validate_startup` currently hard-requires `GOOGLE_API_KEY` — keep that only if some agent resolves to `google`; require `GROQ_API_KEY` when any agent resolves to `groq`. Validation runs via `validate_config()` in `Backend/utils/db.py` called from the `api.py` lifespan.
- **Logging.** `build_agent` logs `supervisor_model`; extend the `extra` dict with `_provider`, and do the same in both sub-agent builders so boot logs show all three resolved pairs.
- **Provider-specific notes.** The `trim_middleware` comment in `supervisor_agent.py` about Gemini rejecting empty contents becomes true again — keep the guard. Gemini system-message handling differs from Groq (`create_agent`'s `system_prompt` is fine; just don't assume Groq-only kwargs like `model_kwargs={"response_format": ...}` port over — that's T15's problem, but the factory should expose a `structured_output(schema)` capability hook now so T15 doesn't reopen this module's interface).
- **Docs.** README's Architecture section ("Supervisor — ... using Google Gemini", and the env-var table showing Groq defaults) must match the new defaults; `.env.example` gains the provider variables.

## Testing Decisions

- Good tests assert the factory's external contract, not construction internals: a Groq setting yields a model bound to the Groq API, a Gemini setting yields one bound to Gemini, an unknown provider raises at startup, and per-agent overrides reach the right agent. All hermetic — no network.
- The factory is the new (and only new) seam; it is deliberately the highest point at which provider choice exists. Prior art: existing settings/config tests and the retrieval-tools factory tests.
- Behavioral acceptance is the scripted eval (fake-MCP seam) run with the new defaults: threshold must hold, and the run must complete with zero rate-limit (429) responses in single-user conditions — the log is the evidence.

## Out of Scope

- Any change to the number of LLM calls per request (T14 owns the fast path). Any prompt changes. Paid-tier selection logic or cost tracking — paid models are reached by setting an environment variable, nothing more.

## Further Notes

Production logs from 2026-07-06 show Groq 429 retry sleeps of 32–56 seconds on routine requests with all three roles sharing one bucket. This PRD is the single largest latency win available without touching architecture.
