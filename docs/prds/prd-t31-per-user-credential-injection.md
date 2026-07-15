# PRD T31 — Per-user credential injection into MCP calls

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T30 (user connectors — storage, encryption, status views; hard dependency). PRD-022 in harmony-retrieval-mcp (per-call `edl_token` contract; **soft** dependency — this PRD feature-detects and degrades cleanly against an un-upgraded MCP, so the two deploy in either order). Phase 2 (consumer side) of the per-user credential series. Decision record 2026-07-14 (design review session).

## Problem Statement

After T30, a researcher's Earthdata token sits encrypted in the database — and nothing reads it. Every satellite-data call still executes as the shared service account: quota attribution, EULA entitlements, and revocation all remain pooled. The delivery step is wiring the stored credential into the one place every model-facing and composite MCP call already passes through — the workspace-binding wrapper that injects `workspace_id` per request — so that a connected user's calls carry their identity, an unconnected (or expired) user's calls fall back to the shared credential, and an old MCP that doesn't know the parameter is simply not sent it.

## Solution

Extend the existing per-call injection seam: when a workspace-bound tool fires, resolve the calling user's `earthdata` connector, decrypt the token just-in-time, and inject it as `edl_token` alongside `workspace_id` — but only when the tool's advertised schema (already fetched and checked at connect time) shows the MCP accepts it, and only when the user has a connected, unexpired token. The parameter is stripped from model-facing schemas exactly like `workspace_id`, so the model never sees or forges identity. New PRD-022 error classes (expired/invalid token, EULA not accepted) surface through the existing deterministic error pipeline as actionable messages, and auth failures flip the connector's stored status so the Connectors tab reflects reality. Successful injected calls update `last_used_at`.

## User Stories

1. As a researcher with a connected Earthdata token, I want every satellite discovery, coverage, and retrieval call to carry my identity to the MCP, so that NASA sees me — my quotas, my EULA acceptances — not a shared account.
2. As a researcher without a connector, I want everything to keep working on the shared service credential exactly as before, so that connecting stays optional, not a new requirement.
3. As a researcher whose token has expired, I want my calls to fall back to the shared credential rather than fail, and my Connectors card to show "expired", so that a lapsed token degrades my attribution, not my access.
4. As a researcher whose token turns out to be revoked/invalid at first use, I want the failure surfaced as a clear "your Earthdata token is invalid — reconnect it in Settings → Connectors" message in chat, so that the fix is one obvious action.
5. As a researcher hitting a EULA-gated collection I haven't licensed, I want the chat error to say "accept the license at Earthdata Login, then retry" (with the resolution link when NASA provides one), so that a per-user entitlement gap reads as my to-do, not a platform bug.
6. As a researcher, I want an auth failure classified against my token to flip my connector's status to error, so that the Connectors tab and the chat error agree about the state of my connection.
7. As a researcher, I want my Connectors card's "last used" to update when my token is actually exercised, so that I can tell my connection is doing something.
8. As a researcher mid-conversation when my token expires, I want long-running jobs that already started with my token to fail with the specific expired-token message (per PRD-022) while new calls degrade to shared, so that the two cases are distinguishable.
9. As a model (agent), I want to never see, receive, or be able to supply `edl_token` — the schema the model sees must not contain it, so identity is uninjectable and unforgeable from the conversation.
10. As a platform operator running an un-upgraded MCP, I want the backend to detect the missing `edl_token` parameter from the advertised schemas and simply not send it, so that this feature deploys in either order with PRD-022 and never trips the connect-time contract check.
11. As a platform operator, I want the decrypted token held only for the duration of call dispatch (resolved just-in-time inside the wrapper, not cached in request state), so that in-process exposure is minimal.
12. As a platform operator, I want the token redacted from every backend log line and error envelope that echoes tool-call arguments, so that per-user credentials never leak through this repo's observability either.
13. As a platform operator, I want per-call connector resolution to be cheap (a short-TTL in-process cache of the encrypted row per user, decryption still just-in-time), so that a multi-tool agent turn doesn't hammer the connectors table.
14. As a developer, I want the injection to live in the same wrapper that injects `workspace_id`, so that there is exactly one seam where calls acquire caller context, not two.
15. As a developer, I want `last_used_at` updates to be fire-and-forget (never on the tool-call critical path, never failing a call), so that bookkeeping can't break retrieval.
16. As a security reviewer, I want the shared-credential fallback to be the *only* silent degradation (never silently substituting one user's token for another's, never falling back mid-job), so that identity semantics stay predictable.

## Implementation Decisions

- **Single seam.** Injection extends the existing workspace-binding wrapper — the one place every model-facing and composite MCP call passes through. It already resolves the current user per-request (context-var getter) and injects `workspace_id`; it now additionally resolves that user's `earthdata` connector, decrypts just-in-time, and adds `edl_token` to the outgoing call kwargs. No second wrapper, no per-tool special cases.
- **Feature detection, not requirement.** `edl_token` is **not** added to the required-parameter contract the connect-time schema check enforces. Instead, at bind time the wrapper records whether each tool's advertised schema includes `edl_token`; only advertising tools get the injection. Against an un-upgraded MCP this feature is inert — no `incompatible` state, no degraded banner, deploy order free (consistent with the degrade-don't-die posture the connection manager already embodies).
- **Injection policy.** Inject iff: user context is bound (the existing missing-context guard already fails loud otherwise) ∧ the user has an `earthdata` connector row ∧ its status is `connected` ∧ `expires_at` is in the future. Otherwise send nothing and let the MCP use its env credential. Expired tokens are never sent — the MCP would only bounce them, and the Connectors tab already renders derived-expired (T30).
- **Model invisibility.** `edl_token` is stripped from model-facing schemas by the same mechanism that strips `workspace_id`. The model cannot see the parameter exists, cannot supply it, and a model-supplied value would be overwritten by the wrapper regardless.
- **Error surfacing and status feedback.** PRD-022's `TOKEN_INVALID`, `TOKEN_EXPIRED`, and `EULA_NOT_ACCEPTED` classifications flow through the existing structured-error pipeline (classification at the wrapper, typed envelopes downstream) and get user-actionable message templates: reconnect-in-settings for invalid, renew for expired, accept-license-with-link for EULA. On `TOKEN_INVALID`/`TOKEN_EXPIRED` attributed to an injected token, the backend flips that connector row's status to `error` (invalid) or relies on derived-expired (expired) so the UI agrees with the failure. EULA failures do not touch connector status — the token is fine; the entitlement is missing.
- **Bookkeeping.** Successful injected calls update the connector's `last_used_at`, fire-and-forget off the critical path, coalesced so one agent turn produces at most one write.
- **Performance.** Per-user connector resolution uses a short-TTL in-process cache of the *encrypted* row (decryption stays just-in-time per call); disconnect/re-paste invalidates by user. No decrypted material is cached anywhere.
- **Secret hygiene.** The wrapper's error paths and any logging of call kwargs redact `edl_token` before write — same obligation PRD-022 takes on the server side, honored independently here so neither repo relies on the other's hygiene.
- **No transport change.** The MCP connection, its static bearer token, and the single shared session are untouched — identity rides per-call in the tool arguments, which is exactly why the single-connection architecture survives this feature.

## Testing Decisions

- The seam under test is the workspace-binding wrapper — the same seam the existing workspace-injection and error-classification tests already exercise; those are the prior art, and fake tools with recorded call kwargs remain the technique. Good tests assert what the MCP would receive and what the user would see, never wrapper internals.
- Injection matrix: connected+unexpired connector + advertising schema → `edl_token` present with the decrypted value; no connector → absent; expired → absent; non-advertising schema → absent even with a valid connector; missing user context → the existing loud failure, unchanged.
- Model invisibility: the model-facing schema for every curated tool contains neither `workspace_id` nor `edl_token`.
- Contract check regression: an MCP whose schemas lack `edl_token` still reaches `ready` (no `incompatible` transition) — the parameter is detected, not required.
- Error classification: envelopes carrying the three new classes render their actionable message templates through the existing error-surfacing path; `TOKEN_INVALID` flips connector status to `error`; `EULA_NOT_ACCEPTED` leaves it untouched.
- Redaction: a wrapper error path with a token in kwargs produces logs/envelopes containing no substring of it.
- Bookkeeping: an agent turn with several injected calls yields one `last_used_at` write; a failing write fails no call.
- Cache: re-paste/disconnect invalidates the resolution cache (no stale token injected within the TTL window).
- Frontend (Vitest): connector card renders the `error` status pushed by a failed injected call; chat error components render the three message templates.
- All via Docker: `docker compose --profile test run --build --rm backend-test` / `frontend-test`. Live verification once both repos are deployed: one account with a real token (confirm attribution/last-used), one without (confirm shared-credential behavior unchanged).

## Out of Scope

- Everything server-side: per-token sessions, job persistence, cache scoping, the error classes' *production* — all PRD-022.
- OAuth, token refresh, or auto-renewal — the injection consumes whatever T30 stored.
- Per-user provenance display ("retrieved as <user>") in the UI — worth a future PRD once identity actually flows.
- Blocking satellite access for unconnected users, or any policy that makes connecting mandatory.
- Sending the token to any surface other than the earthdata MCP tool calls.

## Further Notes

The one deliberate asymmetry: *new* calls from a user with an expired token silently degrade to the shared credential, while an *in-flight job* whose token expires fails loudly with `TOKEN_EXPIRED` (PRD-022 side). That's intentional — at call time we can make an honest choice between identities before anything runs; mid-job substitution would silently change whose entitlements fetched the data after the fact.

If the two repos' deploy order ever matters in practice, it will show up as this feature being inert (schema doesn't advertise the parameter) — which is the designed behavior, not a bug. The place to look is the connect-time schema snapshot, not the connectors table.

## Kickoff

**Recommended model:** Sonnet 5. The wrapper seam, feature-detection machinery, error pipeline, and repository layer all exist with strong prior art; the work is careful wiring, not novel architecture.

**Starter prompt:**
> Implement PRD T31 (`docs/prds/prd-t31-per-user-credential-injection.md`) in Talking-to-Air. T30 must already be merged (user_connectors + Fernet wrapper + status views) — confirm that first, and read T30, T31, and harmony-retrieval-mcp's PRD-022 before starting. Extend the workspace-binding wrapper to feature-detect `edl_token` in each tool's advertised schema at bind time and inject the calling user's decrypted token per the injection policy (connected ∧ unexpired ∧ advertising; otherwise send nothing), strip the parameter from model-facing schemas like `workspace_id`, route the `TOKEN_INVALID`/`TOKEN_EXPIRED`/`EULA_NOT_ACCEPTED` classes through the existing error pipeline with the actionable message templates, flip connector status on invalid, add the coalesced fire-and-forget `last_used_at` write and the short-TTL encrypted-row resolution cache with invalidation on disconnect/re-paste, and redact the token from all logs and envelopes. Do NOT add `edl_token` to the required-parameter contract — an un-upgraded MCP must still reach ready. Write the tests in the PRD's Testing Decisions at the wrapper seam and run both suites via `docker compose --profile test run --build --rm backend-test` and `frontend-test` before considering this done.
