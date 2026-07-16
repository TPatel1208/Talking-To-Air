# PRD T35 — Variable-role taxonomy and dataset inventory

**Repo:** Talking-To-Air · **Session scope:** one session, one commit · **Label:** ready-for-agent
**Depends on:** Nothing hard. Consumes `describe_dataset` (already wrapped in `Backend/services/discovery_service.py:36`) and `collections.yaml`'s `groups:` field. Is the foundation for T36 (companion-variable recommender), which is a pure consumer of this PRD's classification and is **not** built here.

## Problem Statement

A satellite product is not one variable — it is a bundle of them playing different roles: the geophysical product a scientist wants (`vertical_column_troposphere`), the quality variables that say whether to trust it (`main_data_quality_flag`, `*_uncertainty`), viewing geometry, atmospheric/surface context, and internal retrieval/diagnostic fields. Atmospheric scientists don't ask "what category is the variable I'm plotting" — they already know their science product is a science product. They ask **"what else is in this dataset — does it carry cloud fraction, is there a QA flag, can I filter on solar zenith angle?"** That is a dataset-discovery question, and today the app answers none of it.

`describe_dataset` already returns a flat `variables` inventory (`name`/`long_name`/`units`/`advisory_notes`/`mask_note` per variable — see `_compact_variable`, `Backend/earthdata_mcp/workspace.py:215`), but nothing classifies it. `grep -r 'role|category|geolocation|geometry' Backend/datasets` returns zero hits — there is no notion anywhere that `solar_zenith_angle` is geometry and `radiative_cloud_frac` is atmospheric context. The registry's `variables:` field is a *curated Harmony subset*, not a product's true inventory, so it can't be the source of truth either.

The app already models the top two layers of this taxonomy well — science (`primary_var`) and quality (`quality_flag_var`/masking, surfaced via `mask_info.py` and the Metadata tab's trust signal). What's missing is the **semantic inventory**: every variable a product carries, grouped by the role it plays, so a scientist can see at a glance what analyses the dataset makes possible.

## Solution

Add a backend-side classifier — `Backend/datasets/variable_roles.py` — that assigns each variable a **role** from four canonical layers plus an explicit fallback:

- **science** — the geophysical product (requires *positive* evidence)
- **quality** — QA flags, uncertainties, error/std fields
- **context** — atmosphere/surface/geometry bands (cloud fraction, albedo, surface pressure, zenith angles, terrain, aerosol index, PBL)
- **retrieval-metadata** — algorithm intermediates (AMF, slant column, below-cloud partials)
- **unclassified** — the true fallback; never force-fit

Classification combines evidence in strict precedence, mirroring the house style already established by `resolve_mask_info` (`mask_info.py`, override→UMM-Var→CF-attrs) and `identify_time` (`aggregation_service.py:80`, CF-metadata over literal names):

1. **Explicit metadata** — CF `standard_name` / controlled vocabulary (High confidence). Available post-open; often absent from `describe_dataset`'s UMM-Var view.
2. **Group membership** — the HDF group a variable lives in (High/strong). Your files carry this: TEMPO vars are group-qualified (`product/…`), and `collections.yaml` records groups like `key_science_data` and `qa_statistics`. `key_science_data/*` *is* positive evidence for science; `qa_statistics/*` → diagnostic/quality even when the name matches no rule; `geolocation/*` → geometry unambiguously.
3. **Name/marker patterns** — ordered rules, stop on first match (Medium/Low).
4. Otherwise → **unclassified** (None).

Enrich the inventory at the `discovery_service.describe_dataset` seam (a thin passthrough today) — the MCP stays a dumb fact provider; the *semantics* live backend-side next to their consumers, exactly as masking semantics live in `mask_info.py` rather than the MCP. The classified inventory renders as a grouped, name-first view on the dataset-inspection surface, and a lightweight related-variables panel on the chart page.

## User Stories

1. As a scientist inspecting a dataset, I want to see every variable it contains grouped by role (science / quality / context / retrieval-metadata / unclassified), by **name**, so I can tell at a glance what analyses are possible — "does this product carry cloud fraction?" is a name question, not a count question.
2. As a scientist, I want the classifier to say **unclassified** rather than guess, so I never see `fit_residual_rms` or `processing_version` mislabeled as a science product — honest incompleteness over false certainty.
3. As a scientist viewing a chart, I want a small related-variables panel showing the plotted variable's role plus its QA sibling, its uncertainty sibling, and the product's context bands — not a re-render of the whole inventory — so the chart page stays focused while pointing me at what's relevant.
4. As a developer, I want role classification to be a single canonical function consumed by both the inventory UI and (later) T36, so there's one semantic interpretation of a dataset, not two that can drift.
5. As a developer, I want the classifier validated against **real product inventories**, not against the patterns I wrote, so expanding beyond TEMPO doesn't silently regress.

## Implementation Decisions

- **Ground-truth capture is task zero.** Before writing a single classification rule, run `describe_dataset` against every registered collection (`OMI_NO2`, `TROPOMI_NO2`, `TEMPO_NO2`, `TEMPO_O3TOT`, `OMI_O3`, `TEMPO_HCHO`, `TEMPO_HCHO_V03`, `OMI_HCHO`, `MODIS_AOD_TERRA`, `MODIS_AOD_AQUA`) and spot-check with a real granule open where the UMM-Var inventory looks thin. Record the true variable lists as committed fixtures under `Backend/tests/fixtures/variable_inventories/`. The classifier's rule table and the expected-role assertions are built from *these*, not from `collections.yaml`'s `variables:` subset and not from assumed NASA product knowledge. (This corrects examples floated during design — e.g. a `vertical_column_troposphere_uncertainty` band for TEMPO NO₂ — that were extrapolation, not ground truth: the NO₂ registry entry lists only `vertical_column_troposphere` + `main_data_quality_flag`.)
- **`variable_roles.py` API.** A pure `classify_variable(name, *, group=None, standard_name=None, long_name=None, units=None) -> (role, confidence)` and a `classify_inventory(variables, groups) -> list[{name, role, confidence, ...}]` that groups the `describe_dataset` variable list. No I/O, no network — trivially unit-testable, same shape as `mask_info.py`'s pure resolvers.
- **Ordered rules, stop on first match.** Exception markers are checked before science stems, because a science stem is only a default *after* markers miss:
  - quality: `*_uncertainty`, `*_quality_flag`, `*_flag`, `*_error`, `*_std`
  - geometry (context): `*zenith_angle`, `*azimuth*`
  - context: `cloud*`/`*cloud_frac*`, `*surface_pressure`, `albedo`, `terrain*`, `*aerosol_index`, `pbl*`
  - retrieval-metadata: `amf*`, `slant_column*`, `*_below_cloud`
  - science: a recognized geophysical stem (`vertical_column*`, `column_amount*`, `*_aod*`, …) — **only** with positive evidence.
- **Science requires positive evidence; unclassified is the fallback.** A variable is `science` only via a CF `standard_name`, membership in a science group (`product`, `key_science_data`), or a recognized geophysical naming stem. Anything else unmatched → `unclassified`. This keeps `processing_version` / `scan_line` / `orbit_number` / `detector_index` out of the science bucket.
- **Group membership is a strong prior, applied before name patterns.** `qa_statistics/*` → quality/diagnostic; `geolocation/*` → geometry(context); `product`/`key_science_data` → the positive evidence that lets an otherwise-unmarked variable be science. Group signal is available from the group-qualified variable path and `collections.yaml`'s `groups:` list.
- **Four buckets now; nine categories deferred.** The 9 essay sub-categories (geolocation vs geometry vs surface vs atmosphere, diagnostic vs temporal) are a display-only refinement for a later PRD; they don't change behavior and would show mostly-empty groups on 2-variable L3 products.
- **Confidence tiers describe the decision, not the variable:** High (explicit metadata / group), Medium (deterministic marker rule), Low (heuristic keyword), None (unclassified). Surface it so the UI (and later T36) can hedge a Low-confidence guess instead of asserting it.
- **Enrichment seam.** `discovery_service.describe_dataset` attaches a classified `inventory` structure to the result it already returns. Additive — existing callers ignoring the new key are unaffected.
- **Inventory UI (dataset-inspection surface).** Grouped, name-first list with role headers and secondary counts, one clean **Unclassified** group when non-empty, a confidence indicator on Low/None entries. *Dependency to confirm during implementation:* locate the actual describe/preview/inspection surface in the frontend to host this — if no dedicated dataset-inspection panel exists yet, render the inventory in whatever describe/preview surface is present, with the richer home arriving alongside T21/T22.
- **Related-variables panel (chart page).** Plotted variable's role + siblings, matched cheaply: QA sibling = the collection's registry `quality_flag_var`; uncertainty sibling = `<plotted-stem>_uncertainty` if present in the inventory; context/geometry siblings = the inventory's context-role variables. Lightweight — links only, no full inventory, no re-classification.

## Testing Decisions

- Run everything in Docker per CLAUDE.md (`docker compose --profile test run --build --rm backend-test` / `frontend-test`).
- **Golden inventory fixtures (primary).** For each registered collection, a committed fixture of its *real* variable inventory; a table-driven test asserts the classified role for each named variable matches the ground-truth expectation recorded during task zero. This tests against reality, not against the rules — the corpus grows each time a new product surprises the classifier.
- **Collision / negative units (the sharp cases):** `vertical_column_troposphere_uncertainty` → quality (not science); `vertical_column_stratosphere` → science; `processing_version` / `scan_line` / `orbit_number` → unclassified (proving science-as-default is gone); `amf_troposphere` → retrieval-metadata.
- **Group-prior units:** a variable in `qa_statistics/` with an unrecognized name → quality/diagnostic; `geolocation/solar_zenith_angle` → geometry(context); a bare-name variable in `key_science_data/` → science.
- **Confidence assertions:** a `standard_name`/group hit reports High; a marker-rule hit reports Medium; an unmatched variable reports None + `unclassified`.
- **Enrichment test:** `discovery_service.describe_dataset` returns the additive `inventory` structure without disturbing existing keys.
- **Frontend:** the inventory panel renders role groups with names from a fixture; a fixture with an unclassified variable renders the Unclassified group; the chart-page related-variables panel renders the plotted role + siblings and renders nothing spurious when a product has no context bands (e.g. MODIS AOD).
- **Live verification:** open a TEMPO O₃ dataset (the one product with real context bands — `radiative_cloud_frac`, `uv_aerosol_index`, `o3_below_cloud`) and confirm the inventory groups them as context/retrieval-metadata; open MODIS AOD and confirm it shows science + unclassified only, no invented context. Use the `test`/`1234` account per CLAUDE.md.

## Out of Scope

- **All of T36** — companion-variable recommender, suggested actions, evidence synthesis, on-demand narrative. T35 only produces the classification T36 will consume.
- **The 9-category display refinement** — deferred; four buckets ship here.
- **Any retrieval change** — T35 is metadata-only, classifying the `describe_dataset` inventory. It does not widen the `variables:` subset or pull extra bands.
- **Chart-overlay of context variables** — that's T29 (unbuilt) / T36, not here.
- **Pushing classification into `harmony-retrieval-mcp`** — deliberately kept backend-side; the MCP stays a fact provider.

## Further Notes

The taxonomy is an **L3-honest** version of an L2 mental model. The essay that motivated this describes L2 swath richness (per-pixel solar/viewing zenith, surface pressure, terrain, PBL); most collections here are L3 gridded aggregates where those bands simply don't exist. That's *why* `unclassified` and mostly-empty context groups are correct outputs, not bugs — the inventory truthfully showing "this product has no geometry bands" is useful information. If the app later serves L2 granules (cf. T21 granule inspection), the same classifier applies unchanged to a much richer inventory.

The related-variables panel is intentionally the thin edge of T36: it's the one chart-page affordance that points at companion variables, built here from the classification directly so the chart page gets immediate value without waiting for the full recommender.

## Kickoff

**Recommended model:** Opus 4.8. The hard part isn't code volume — it's judgment: capturing ground truth from live inspection, and designing an ordered-evidence classifier that is conservative (science needs positive evidence) and correct across products it wasn't tuned on. The live ground-truth step needs the `earthdata-mcp` connected and the `test`/`1234` account.

**Starter prompt:**
> Implement PRD T35 (`docs/prds/prd-t35-variable-role-taxonomy.md`) in Talking-to-Air. **Do task zero first:** run `describe_dataset` against every collection registered in `Backend/datasets/collections.yaml` (spot-checking with a real granule open where UMM-Var looks thin), and commit the true variable inventories as fixtures under `Backend/tests/fixtures/variable_inventories/`. Build the classifier and its expected-role table from those fixtures — not from the registry `variables:` subset, which is a curated Harmony subset, not ground truth. Then add `Backend/datasets/variable_roles.py`: a pure `classify_variable(...) -> (role, confidence)` over four canonical roles (science / quality / context / retrieval-metadata) plus `unclassified`, using ordered evidence — explicit CF `standard_name` and group membership first (High), deterministic marker rules next (Medium/Low), unclassified as the true fallback. Science must require positive evidence (CF standard_name, a science group like `product`/`key_science_data`, or a recognized geophysical stem) — never the residual default; `processing_version`/`scan_line`/`orbit_number` must classify as unclassified. Mirror the precedence style of `datasets/mask_info.py`'s `resolve_mask_info`. Attach a classified `inventory` to the result of `Backend/services/discovery_service.py`'s `describe_dataset` (additive; MCP stays untouched). Add the grouped, name-first inventory UI on the frontend dataset-inspection surface (locate it first; fall back to the existing describe/preview surface if no dedicated panel exists), and a lightweight related-variables panel on the chart page (QA sibling = registry `quality_flag_var`; uncertainty sibling = `<stem>_uncertainty`; context siblings = inventory context vars). Write the golden-inventory, collision/negative, group-prior, and confidence tests per Testing Decisions and run both suites via `docker compose --profile test run --build --rm backend-test` and `frontend-test`. Live-verify with TEMPO O₃ (has real context bands) and MODIS AOD (has none) using the `test`/`1234` account. Do not implement any part of T36.
