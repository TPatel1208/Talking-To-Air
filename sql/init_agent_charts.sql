CREATE TABLE IF NOT EXISTS agent_charts (
    -- TEXT, not UUID: chart ids are the same value the code stores as
    -- `chart_id` and the frontend cites back to look a chart up. Generic
    -- charts get a uuid5 content hash, but T06 artifact-typed plots
    -- (map/comparison/timeseries) mint human/LLM-readable prefixed ids like
    -- `map_52fd40b2e418`, which a UUID column rejects at insert time.
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '__legacy__',
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE agent_charts
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '__legacy__';

-- Widen pre-existing volumes whose id column was created as UUID (safe/no-op
-- once it is already TEXT). Without this, prefixed artifact ids fail to persist.
ALTER TABLE agent_charts
    ALTER COLUMN id TYPE TEXT USING id::text;

CREATE INDEX IF NOT EXISTS idx_agent_charts_thread_created
ON agent_charts (thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_charts_user_id
ON agent_charts (user_id);
