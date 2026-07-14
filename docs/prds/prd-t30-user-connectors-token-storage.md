# PRD T30 — User connectors: per-user Earthdata token storage and Connectors UI

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** nothing external — self-contained in this repo. Phase 1 of the per-user credential series (T30 → PRD-022 in harmony-retrieval-mcp → T31). Decision record 2026-07-14 (design review session).

## Problem Statement

Every user of the platform reaches NASA Earthdata through one shared service credential that lives in the harmony-retrieval-mcp stack's environment. Individual researchers have their own Earthdata Login (EDL) accounts — with their own EULA acceptances, quota attribution, and revocation control — but the platform has nowhere to hold a per-user credential: there is no secret storage of any kind in this codebase (bcrypt password hashing is one-way and unusable for this), no UI to connect an external account, and no data model tying a user to an external service credential. Until that exists, per-user NASA access is impossible, and adding any future per-user external service (EPA AQS keys, etc.) would hit the same wall.

## Solution

A "Connectors" system: a per-user, encrypted credential store keyed off the existing `users` table, a small CRUD API that never returns the secret, and a Connectors tab in the frontend where a user pastes their self-service EDL user token (generated at urs.earthdata.nasa.gov → "Generate Token"; valid 60 days, max 2 concurrent per EDL account). The stored token is not consumed by anything yet — retrieval stays on the shared credential until PRD-022/T31 land — but this phase establishes the storage, encryption, API, and UI that those phases plug into, with no external dependency (no OAuth app registration, no MCP contract change).

## User Stories

1. As a researcher, I want a Connectors tab in the app, so that I can see which external services my account can be linked to and their current connection state.
2. As a researcher, I want to paste my Earthdata Login user token into the Earthdata connector card, so that the platform can later act against NASA services as me rather than as a shared service account.
3. As a researcher, I want the paste field to behave like a password field (masked, not echoed back after save), so that my token is never displayed on screen after entry.
4. As a researcher, I want immediate feedback if the token I pasted is malformed or already expired, so that I don't discover a bad token days later when a retrieval fails.
5. As a researcher, I want the connector card to show when my token expires, so that I can renew it before it lapses (EDL tokens last 60 days).
6. As a researcher, I want the card to visibly flip to an "expired" state once the expiry passes, so that a stale connection is impossible to mistake for a live one.
7. As a researcher, I want a Disconnect action that deletes my stored token, so that I can sever the link at any time (and I know I can also revoke the token itself at Earthdata Login, since it's mine).
8. As a researcher, I want re-pasting a new token to replace the old one in place, so that renewal is one action, not disconnect-then-reconnect.
9. As a researcher, I want a link on the connector card to the EDL page where tokens are generated, so that I don't have to hunt for where "my token" comes from.
10. As a researcher, I want assurance (stated in the UI) that my token is stored encrypted and is never shown to anyone including me after save, so that I can trust the platform with it.
11. As a user who has not connected anything, I want each connector card to show a clear "not connected" state with a one-line explanation of what connecting does, so that the tab is self-explanatory.
12. As a platform operator, I want the secret encrypted at rest with a key held only in the backend's environment, so that a database dump alone cannot expose any user's token.
13. As a platform operator, I want key rotation to be possible (accept old + new key simultaneously), so that a suspected key exposure doesn't force every user to reconnect.
14. As a platform operator, I want the connectors feature to degrade cleanly when the encryption key isn't configured (endpoints answer "not configured on this server", boot proceeds), so that ground/EPA-only deployments aren't forced to configure a feature they don't use.
15. As a platform operator, I want the API to never return the encrypted or decrypted secret in any response, so that the token cannot leak through the frontend, browser devtools, or logs of API traffic.
16. As a security-conscious operator, I want no raw password custody anywhere in this feature — tokens only, never EDL usernames/passwords — so that a breach exposes only revocable, expiring tokens.
17. As a future developer, I want the data model to be connector-type-generic (a registry of connector types drives the UI), so that adding a second connector type later is a registry entry plus its auth specifics, not a schema redesign.
18. As a future developer (T31), I want the stored row to carry `last_used_at` and a decodable expiry, so that credential injection can update usage and skip expired tokens without schema changes.
19. As a user of a deployment where the feature is unconfigured, I want the Connectors tab to say so plainly instead of erroring, so that I know it's a server-side choice, not a bug.
20. As an operator auditing the codebase, I want the now-dead `EDL_USERNAME`/`EDL_PASSWORD` settings fields and env examples removed, so that nobody mistakes them for the mechanism this feature uses.

## Implementation Decisions

- **Data model.** New table `user_connectors` in this stack's Postgres, created at boot via the existing repository `CREATE TABLE IF NOT EXISTS` pattern (same lifecycle as the `users` and `revoked_tokens` tables — no migration framework exists and none is introduced). Columns: primary key, `user_id` FK → users, `connector_type` (text, e.g. `earthdata`), `auth_method` (text; only `token` in this phase — `oauth` is a future value, `credentials`/password is deliberately not a value at all), `encrypted_secret` (text — Fernet output is already urlsafe-base64, so no bytea), `expires_at` (timestamptz, decoded from the token), `status` (text: `connected` | `revoked` | `error`; **expired is derived from `expires_at`, never stored**), `connected_at`, `last_used_at` (nullable; written by T31, only provisioned here). Unique constraint on `(user_id, connector_type)`; upsert on re-paste.
- **Encryption.** New dependency `cryptography`. `MultiFernet` from day one: the new `CONNECTOR_ENCRYPTION_KEY` env var accepts a comma-separated key list (first = encrypt, all = decrypt) so rotation is re-encrypt-on-read or lazy, never a flag day. Decryption happens only in-process; the decrypted value never crosses the API boundary.
- **Feature gating, degrade-don't-die.** If `CONNECTOR_ENCRYPTION_KEY` is unset, boot proceeds (consistent with the T17 philosophy that optional capabilities never block ground/EPA use); connector endpoints return a structured 503 "connectors not configured on this deployment", and the frontend renders that state. If the key is set but malformed, startup validation fails loudly (a half-configured secret store is worse than none).
- **Token validation on paste.** The pasted EDL user token is a JWT: decode locally (no signature verification against EDL — we are not the audience), require a parseable `exp` in the future, reject otherwise with a specific message. No live round-trip to EDL in the request path — liveness is proven at first real use (PRD-022/T31 classify auth failures and flip `status` to `error`). This keeps the paste flow dependency-free and fast.
- **API contract.** Endpoints under the authenticated API (same JWT middleware / `request.state.current_user` scoping as every other per-user endpoint): list connectors (registry entries merged with the current user's rows — returns `connector_type`, `display_name`, `auth_method`, `status` including the derived `expired`, `connected_at`, `expires_at`, docs URL; **never any secret material, encrypted or not**), set/replace the token for a connector type (body: the raw token; response: the new status view), and disconnect (delete the row). All responses are status views; there is no read-secret endpoint by design.
- **Connector registry.** A static backend-config list of `{connector_type, display_name, auth_method, token_docs_url, description}` — one entry (`earthdata`) now. The list endpoint is registry-driven so the frontend renders cards generically; adding a connector type later is a registry entry.
- **Frontend.** A Connectors tab/section in the existing account-settings surface: one card per registry entry showing display name, status badge (not connected / connected until <date> / expired / error / server-not-configured), a masked paste field with save, a Disconnect action, a "Generate a token at Earthdata Login" link, and a one-line "stored encrypted, never displayed after save" notice. Respect the repo's strict react-hooks lint rules (no setState-in-effect; prop-driven state resets).
- **Dead-config cleanup.** Remove the unused `edl_username`/`edl_password` settings fields and the `EDL_USERNAME`/`EDL_PASSWORD` lines from the env example and README env table — they are read by nothing in this repo and misleadingly suggest a password-based mechanism this feature explicitly avoids.
- **Explicitly not in this phase:** nothing consumes the stored token yet. `bind_workspace` and the MCP client are untouched. The shared service credential in the MCP stack keeps doing 100% of NASA-facing auth until PRD-022 and T31.

## Testing Decisions

- Tests assert external behavior at two seams, both pre-existing: the **repository layer** (schema contract + CRUD semantics, prior art: the chart repository's schema-contract test and other repository tests) and the **API route layer** (authenticated request in → status view out, prior art: the owned-chart and session endpoints' per-user scoping tests).
- Crypto round-trip: a stored secret decrypts to what was pasted; a secret encrypted under an old key still decrypts after a new key is prepended (MultiFernet rotation); a row is never readable without the key.
- Secret-never-leaves: every connector endpoint response is asserted to contain no substring of the pasted token, encrypted or raw. This is the security property, tested as behavior.
- Paste validation: expired JWT rejected with the specific reason; non-JWT garbage rejected; valid future-dated token accepted and `expires_at` matches the token's `exp`.
- Derived-expired: a row whose `expires_at` is past reports `expired` in the list view without any stored status change.
- Feature gating: with no `CONNECTOR_ENCRYPTION_KEY`, endpoints return the structured 503 and boot-time validation passes; with a malformed key, startup validation fails.
- Per-user isolation: user A's connector never appears in user B's list; disconnect only deletes the caller's row.
- Frontend (Vitest): card renders each status variant from the API view; paste field masks input; save/disconnect call the right endpoints; not-configured state renders the server-side-choice message.
- Everything runs through Docker per the repo rule: `docker compose --profile test run --build --rm backend-test` / `frontend-test`.

## Out of Scope

- Consuming the token anywhere (MCP calls, retrieval) — that is PRD-022 (MCP side) and T31 (injection).
- EDL OAuth (authorize/callback/refresh). The data model reserves `auth_method='oauth'` but no OAuth code lands here; app registration with ESDIS is approval-gated and off the critical path.
- Storing EDL usernames/passwords — permanently out of scope, not just deferred.
- Live token verification against EDL at paste time.
- Admin views of other users' connectors; connector audit logs.
- Any change to the harmony-retrieval-mcp repo or the shared-credential fallback.

## Further Notes

EDL user tokens are self-service, last 60 days, and each account may hold at most 2 concurrent tokens — the paste flow occupies one of the user's two slots, which is fine, but the UI copy should say "generate a token" not "create a new account/app". The 60-day lifetime is why `expires_at` and the derived-expired state are first-class in this phase: renewal UX is the main ongoing friction of the token-paste approach, and it's what OAuth would eventually remove.

The encryption key ships as a generated value (`Fernet.generate_key()`) documented in the env example. Losing the key strands every stored token (users just reconnect); that trade-off is acceptable and should be stated in the env example comment.

## Kickoff

**Recommended model:** Sonnet 5. The design is fully specified, the patterns (repository, per-user endpoint scoping, settings validation) all have in-repo prior art, and there is no architecturally novel piece.

**Starter prompt:**
> Implement PRD T30 (`docs/prds/prd-t30-user-connectors-token-storage.md`) in Talking-to-Air. Read the PRD fully first. Build the `user_connectors` repository (CREATE TABLE IF NOT EXISTS pattern, unique on user_id+connector_type), the MultiFernet secret wrapper keyed from `CONNECTOR_ENCRYPTION_KEY` (comma-separated keys, degrade-to-503 when unset, fail startup when malformed), the registry-driven connector endpoints (list/set-token/disconnect, JWT-scoped via request.state.current_user, no response ever contains secret material), JWT paste validation (decode exp locally, no EDL round-trip), and the Connectors tab with the card states described. Remove the dead `edl_username`/`edl_password` settings fields and env-example lines. Write the tests in the PRD's Testing Decisions at the repository and route seams, and run both suites via `docker compose --profile test run --build --rm backend-test` and `frontend-test` before considering this done. Do not touch `bind_workspace`, the MCP client, or the harmony-retrieval-mcp repo.
