"""Verify that the satellite agent's cacheable prompt prefix stays stable.

The earthdata agent is stateless, so every call re-sends its system prompt and
tool schemas -- roughly 12,000 tokens, about 86% of a typical call's input.
Gemini serves that prefix back at a discounted rate through implicit prompt
caching, which requires the prefix to be byte-identical from one call to the
next.

Nothing configures that discount, so nothing protects it either. Rendering a
timestamp, a uuid or a thread id into the prompt would drop the cache hit rate
to zero, no other test would fail, and the only symptom would be a larger bill.
These tests are that protection.

Scope: this covers the parts built in this repository -- the system prompt and
the handle tools. The curated tools' schemas come from the earthdata MCP server
and are not this repository's to pin.
"""
import datetime
import json
import re
import unittest


def _tool_schemas() -> str:
    """The handle tools' schemas, serialized the way a provider sees them.

    Uses ``_handle_tools`` rather than ``build_satellite_tools`` because the
    latter requires a populated ``mcp_tools`` dict, and stubbing that dict
    would only test the stubs.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from tta_backend.tools.satellite_tools.factory import _handle_tools

    return json.dumps([convert_to_openai_tool(tool) for tool in _handle_tools({})], sort_keys=True)


class SystemPromptIsByteStableTests(unittest.TestCase):
    def test_two_builds_of_the_prompt_are_byte_identical(self):
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        self.assertEqual(get_earthdata_agent_prompt(), get_earthdata_agent_prompt())

    def test_the_prompt_does_not_embed_the_current_date(self):
        """The agent is told the current date through the per-task
        ``[Current date/time: ...]`` banner, which sits after the prefix.
        Rendering it into the system prompt instead would put changing text in
        front of every cached byte.
        """
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prompt = get_earthdata_agent_prompt()
        now = datetime.datetime.now(datetime.timezone.utc)
        for rendered in (
            now.strftime("%Y-%m-%d"),
            now.strftime("%d/%m/%Y"),
            now.strftime("%m/%d/%Y"),
            now.strftime("%B %d, %Y"),
            now.strftime("%b %d %Y"),
        ):
            self.assertNotIn(rendered, prompt, f"today's date is rendered into the system prompt as {rendered!r}")

    def test_the_prompt_carries_no_per_request_identifier(self):
        """A thread id, workspace id or request id reaching the prompt would
        give each user a different prefix instead of one shared between them.
        A uuid is what all three look like.
        """
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        uuid_like = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        found = uuid_like.search(get_earthdata_agent_prompt())
        self.assertIsNone(found, f"system prompt carries a uuid-shaped identifier: {found and found.group(0)!r}")


class ToolSchemasAreByteStableTests(unittest.TestCase):
    def test_two_builds_of_the_handle_tools_serialize_identically(self):
        self.assertEqual(_tool_schemas(), _tool_schemas())

    def test_the_schemas_do_not_depend_on_the_mcp_tools_dict_identity(self):
        """The tools close over ``mcp_tools`` and index it at call time, and
        the agent is rebuilt with a fresh dict whenever the MCP reconnects. A
        description or argument schema derived from that dict would change the
        prefix on every reconnect.
        """
        from langchain_core.tools import BaseTool
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from tta_backend.tools.satellite_tools.factory import _handle_tools

        class _Stub(BaseTool):
            name: str = "stub"
            description: str = "stub"

            def _run(self, *args, **kwargs):  # pragma: no cover -- never invoked
                return ""

        def schemas(mcp_tools):
            return json.dumps(
                [convert_to_openai_tool(tool) for tool in _handle_tools(mcp_tools)], sort_keys=True
            )

        self.assertEqual(schemas({}), schemas({"export_result": _Stub(), "rematerialize": _Stub()}))

    def test_the_prefix_clears_the_size_the_cache_discount_needs(self):
        """Implicit caching only applies above a provider minimum; the hit
        quantum measured on this workload was about 4,050 tokens. Shrinking
        the prefix below one block would stop it being cached at all, which
        would raise spend while looking like a saving.

        Measured in characters rather than tokens so this does not become a
        test of a tokenizer. 4 chars/token is the approximation already used
        elsewhere in this repository.
        """
        from tta_backend.config.earthdata_agent_prompt import get_earthdata_agent_prompt

        prefix_chars = len(get_earthdata_agent_prompt()) + len(_tool_schemas())

        self.assertGreater(prefix_chars // 4, 4_050, "cacheable prefix has fallen below one ~4,050-token cache block")


if __name__ == "__main__":
    unittest.main()
