# PRD series — research-workbench evolution (TTA side)

One session, one commit each. The MCP-side prerequisites (PRD 016–019) live in
the harmony-retrieval-mcp repo under `docs/prds/`.

| PRD | Title | Depends on |
|---|---|---|
| [T01](prd-t01-consolidation-compose-topology.md) | Consolidate on v2 + joint compose topology | MCP 018 |
| [T02](prd-t02-mcp-client-toolset-composites.md) | MCP client, curated toolset, composites | T01, MCP 017+018 |
| [T03](prd-t03-open-handle-plot-stat-rework.md) | `open_handle` + plot/stat rework + loader deletion | T02 |
| [T04](prd-t04-agent-prompts-structured-returns-eval.md) | Agent prompts, structured returns, models, eval | T03 |
| [T05](prd-t05-jobs-panel.md) | Jobs panel | T02, T04, MCP 019 (`list_workspace`) |
| [T06](prd-t06-artifact-generalization.md) | Artifact types: map, comparison, timeseries | T03, T05 |
| [T07](prd-t07-satellite-ground-validation.md) | Satellite↔ground validation workflow | T06 |
| [T08](prd-t08-region-period-comparison.md) | Region/period comparison workflow | T06 |
| [T09](prd-t09-discovery-pane-gibs-quicklook.md) | Discovery pane + GIBS quick-look | T02, T05 |
| [T10](prd-t10-provenance-citations-exports.md) | Provenance pane, citations, methods & data export | T06, T07/T08 |
| [T11](prd-t11-agent-contract-repairs.md) | Contract repairs + legacy tool removal — MCP-first minimal toolset | T10 committed |
| [T12](prd-t12-provider-split-model-factory.md) | Provider split — supervisor on Gemini Flash via model factory | T11 |
| [T13](prd-t13-tool-result-compaction.md) | Tool-result compaction + subagent trim safety net | T11 |
| [T14](prd-t14-router-fast-path.md) | Deterministic router fast path + history write-back | T12, T13 |
| [T15](prd-t15-envelope-enforcement-salvage.md) | Envelope enforcement, salvage, cheap retry | T11, T12 |
| [T16](prd-t16-eval-latency-budgets-offload.md) | Eval hardening — latency budgets, e2e routing tasks, event-loop offload | T12–T15 |

Cut line (decision record 2026-07-04): Phase 4 (projects/multi-user) slides
first, then T10's pane; T07/T08/T09's workflows never slide. The MCP live
matrix (MCP PRD 019) runs concurrently with T01–T04.

**Tracker note:** intended as GitHub issues labeled `ready-for-agent`; blocked
on the PAT lacking Issues write permission — publish these bodies as issues
once fixed.
