# PRD T09 — Dataset discovery pane + GIBS quick-look

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** T05 (workbench layout), T02 (client). Third signature workflow's UI.

## Problem Statement

As a researcher, discovering data currently means asking the chat and reading prose. I can't browse candidates, compare their variables and coverage side by side, or confirm "is this the right product over my region?" without committing to a minutes-long retrieval. Discovery is open-ended in the data layer now, but the experience is still a keyhole.

## Solution

The third workbench pane: a dataset discovery pane. A search box drives the MCP's dataset search; each result renders as a card (description summary, variables, temporal extent, provider) with three actions — GIBS quick-look (an inline browse image in seconds, no retrieval), coverage check for the current AOI/window, and retrieve (which hands off to the chat/agent flow). The agent uses the same quick-look before any retrieval it initiates, so both entry points share the confirm-before-commit habit.

## User Stories

1. As a researcher, I want to type a phenomenon ("soil moisture", "formaldehyde column") into a search pane and see dataset cards, so that I can survey candidates without conversational back-and-forth.
2. As a researcher, I want each card to show the dataset's summary, key variables, temporal extent, and provider, so that I can shortlist without opening anything.
3. As a researcher, I want a quick-look button that renders a GIBS browse image inline for my area of interest, so that I confirm product-and-region fit in seconds instead of minutes.
4. As a researcher, I want a coverage check from the card for my current AOI and time window, so that "is there data when/where I need it" is one click.
5. As a researcher, I want a retrieve action on the card that starts the standard agent retrieval flow (safe_retrieve gates included), so that browsing connects to the same guarded pipeline as chat.
6. As the earthdata agent, I want to call the dataset preview before any retrieval I plan and have the browse image render inline in chat, so that the researcher confirms before the platform commits resources.
7. As a researcher whose dataset has no GIBS layer, I want the quick-look to say so plainly rather than show nothing, so that absence of preview isn't mistaken for absence of data.

## Implementation Decisions

- Backend: thin authenticated proxy endpoints over the MCP's search/describe/preview/coverage tools for the pane's direct (non-agent) use — same workspace binding and token handling as the agent path, so the pane cannot do anything the agent couldn't.
- The GIBS URL from the preview tool is rendered client-side as an image; no imagery passes through the backend.
- Card retrieve action posts a structured prompt into the chat (the agent runs its normal flow) rather than bypassing the agent — one retrieval pipeline, two entry points.
- Pane state (query, results, selected AOI/window context) is client-side; the AOI/window used by card actions is the session's current one, shown on the pane so the researcher knows what coverage/quick-look are evaluated against.

## Testing Decisions

- Backend proxy endpoints tested at the endpoint seam against the fake MCP: search/describe/coverage passthrough, workspace scoping, and the no-GIBS-layer answer shape.
- Frontend demo-verified per the standing decision; exit script: search "soil moisture" → card → quick-look renders → coverage check → retrieve → job appears in the jobs panel (T05).

## Out of Scope

- Provenance pane (T10). Saved searches/collections-of-interest (Phase 4 projects territory).
- Any new MCP capability — the pane consumes existing discovery tools only.

## Further Notes

Completes the visible surface of signature workflow #3 (open-ended discovery). The previously-impossible demo task ("find a soil-moisture dataset and map it over the Raritan basin") should now be doable either conversationally or by browsing.
