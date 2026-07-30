CREATE TABLE IF NOT EXISTS agent_artifacts (
    -- TEXT, not UUID: artifact ids are ArtifactStore's own `tbl_<hex12>`
    -- ids, the same value the frontend cites back to fetch a page or export
    -- a CSV. A UUID column would reject them at insert time.
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    title TEXT NOT NULL,
    columns JSONB NOT NULL,
    rows JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- NULL until claimed. Rows are only ever written at claim time (mint
    -- stays in-memory-only), so this is always set today -- kept nullable so
    -- the unclaimed-row sweep stays correct if a future write path changes
    -- that.
    claimed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_artifacts_thread_created
ON agent_artifacts (thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_artifacts_user_id
ON agent_artifacts (user_id);

CREATE INDEX IF NOT EXISTS idx_agent_artifacts_unclaimed
ON agent_artifacts (created_at)
WHERE claimed_at IS NULL;
