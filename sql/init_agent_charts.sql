CREATE TABLE IF NOT EXISTS agent_charts (
    id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '__legacy__',
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE agent_charts
    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT '__legacy__';

CREATE INDEX IF NOT EXISTS idx_agent_charts_thread_created
ON agent_charts (thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_charts_user_id
ON agent_charts (user_id);
