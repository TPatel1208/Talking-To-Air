CREATE TABLE IF NOT EXISTS agent_charts (
    id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_charts_thread_created
ON agent_charts (thread_id, created_at);
