"""Agent steps in the workflow runner: what the model sees, the typed answer, retries, failures and the trace."""
import dataclasses
import datetime
import json
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.test_harness_sources_satellite import NO2
from tests.test_harness_workflow_runner import DATASETS, SECRET, Fixture, Script, fake_registry, make, spec
from tta_backend.harness.capabilities.failures import MODEL_ANSWER_INVALID, MODEL_FAILED
from tta_backend.harness.knowledge import Note, NoteCatalogue
from tta_backend.harness.models import NO_BACKOFF, ModelChoice, NamedModel
from tta_backend.harness.results import Gap, Result
from tta_backend.harness.run_context import NO_MODEL_REASON
from tta_backend.harness.testing import ScriptedChatModel, ScriptedModels, answer, env_factory, rate_limited, tool_call
from tta_backend.harness.trace import MAX_JSON_BYTES, FailureCategory, TraceEntry
from tta_backend.harness.workflows import RetrySettings, WorkflowRunner
from tta_backend.harness.workflows import agent as agent_module
from tta_backend.harness.workflows.agent import ANSWER_TOOL, MAX_READ_BYTES, MAX_READS_BYTES, MAX_TEXT_CHARS
from tta_backend.harness.workflows.compile import AGENT_INTERNAL, ANSWER_NOT_RECORDED
from tta_backend.harness.workflows.loader import load_specs

INSTRUCTION = "Say what the totals show and what they do not."
SHAPE = (f"text (text, at most {MAX_TEXT_CHARS} characters); ok (true or false)")
SYSTEM = (
    f"You carry out one step of a workflow. {INSTRUCTION}\n\n"
    "Use only the values and notes you are given; you cannot fetch data or call other tools. "
    f"Give your answer by calling {ANSWER_TOOL} once, with these fields: {SHAPE}."
)
CORRECTION = f"That answer did not fit the declared shape. Call {ANSWER_TOOL} once with exactly these fields: {SHAPE}."

NOTES = NoteCatalogue([
    Note(id="tempo-hours", title="TEMPO hours", body="TEMPO sees daylight hours only.", datasets=("TEMPO_NO2",)),
    Note(id="column-not-surface", title="Columns", body="A column is not a surface concentration.", quantities=(NO2,)),
    Note(id="ozone-layers", title="Ozone layers", body="Unrelated.", datasets=("TROPOMI_O3",)),
])


def interpret(**over: Any) -> dict[str, Any]:
    agent = {"instruction": INSTRUCTION, "reads": ["$total.summary", "$base.product", "$quantity"],
             "returns": {"text": "string", "ok": "boolean"}}
    agent.update(over.pop("agent", {}))
    return {"id": "interpret", "agent": agent, **over}


def agent_spec(*extra: dict[str, Any], agent: dict[str, Any] | None = None, outputs: dict[str, Any] | None = None,
               **over: Any) -> dict[str, Any]:
    """Two use steps, then an agent step reading them, then whatever else is given."""
    return spec(
        [
            make("base", quantity="$quantity", value=2),
            make("total", a="$base"),
            agent if agent is not None else interpret(),
            *extra,
        ],
        outputs or {"data": ["$total"], "interpretation": "$interpret.text"},
        inputs={"quantity": {"type": "quantity", "ask": "Which quantity?"}},
        products={"quantity": ["TEMPO_NO2", "TROPOMI_NO2"]},
        **over,
    )


def said(text: str = "The total is 3.", ok: bool = True, usage: tuple[int, int] | None = (12, 7)) -> AIMessage:
    return tool_call(ANSWER_TOOL, {"text": text, "ok": ok}, usage=usage)


def choice(*models: tuple[ScriptedChatModel, str]) -> ModelChoice:
    first, *rest = (NamedModel(model, name) for model, name in models)
    return ModelChoice(first, tuple(rest))


class AgentFixture(Fixture):
    """A runner fixture whose runs have models, and whose runner may have notes."""

    def __init__(self, tmp: str, registry: Any, catalogue: Any, script: Script, *, models: Any,
                 notes: NoteCatalogue | None = None, **kw: Any) -> None:
        super().__init__(tmp, registry, catalogue, script, **kw)
        self.notes = notes
        self.build = env_factory(tmp, earthdata=script.earthdata, places=self.places, models=models)
        self.runner = self.new_runner()

    def new_runner(self) -> WorkflowRunner:
        return WorkflowRunner(
            self.registry, self.catalogue, self.saver,
            retry=RetrySettings(retries=2, backoff=lambda n: float(n), sleep=self.sleeps_into, model_backoff=NO_BACKOFF),
            notes=getattr(self, "notes", None),
        )

    async def entry(self, run_id: str, step: str) -> TraceEntry:
        [found] = [e for e in await self.stores.traces.entries(self.context.identity.user_id, run_id) if e.step == step]
        return found


class AgentCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.script = Script()
        self.registry = fake_registry(self.script)
        self.model = ScriptedChatModel(script=[said()])

    def fixture(self, source: dict[str, Any], *, models: Any = "default", notes: NoteCatalogue | None = NOTES,
                **kw: Any) -> AgentFixture:
        catalogue = load_specs({"spec": source}, self.registry, DATASETS)
        if models == "default":
            models = ScriptedModels({"agent_step": choice((self.model, "fake:primary"))})
        return AgentFixture(self.tmp, self.registry, catalogue, self.script, models=models, notes=notes, **kw)

    async def start(self, f: AgentFixture) -> Result:
        return await f.start({"quantity": NO2})


class WhatTheModelSeesTests(AgentCase):
    async def test_the_model_gets_only_the_instruction_the_values_it_reads_and_the_notes(self):
        f = self.fixture(agent_spec())
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)

        [sent] = self.model.received
        human = "\n".join([
            "Values:",
            '$total.summary: {"value": 3.0}',
            '$base.product: "TEMPO_NO2"',
            f'$quantity: "{NO2}"',
            "",
            "Notes:",
            "Columns: A column is not a surface concentration.",
            "TEMPO hours: TEMPO sees daylight hours only.",
        ])
        self.assertEqual([(type(m), m.content) for m in sent], [(SystemMessage, SYSTEM), (HumanMessage, human)])
        # No tools but the one that carries the answer.
        self.assertEqual(self.model.tools_offered, [[ANSWER_TOOL]])

    async def test_with_prompt_caching_the_system_prompt_carries_a_breakpoint(self):
        cached = ModelChoice(NamedModel(self.model, "fake:primary"), prompt_cache=True)
        f = self.fixture(agent_spec(), models=ScriptedModels({"agent_step": cached}))
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)

        [[system, human]] = self.model.received
        self.assertIsInstance(system, SystemMessage)
        self.assertEqual(system.content, [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}])
        # Only the system prompt is marked in an agent step.
        self.assertIsInstance(human.content, str)
        saved = repr(f.saver.storage) + repr(f.saver.writes) + repr(f.saver.blobs)
        self.assertNotIn("cache_control", saved)

    async def test_without_notes_the_model_is_given_none(self):
        f = self.fixture(agent_spec(), notes=None)
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)
        [[_, human]] = self.model.received
        self.assertNotIn("Notes:", human.content)
        self.assertEqual((await f.entry(result.trace_id, "interpret")).params["notes"], [])

    async def test_the_agent_run_stays_out_of_the_workflow_checkpoint(self):
        f = self.fixture(agent_spec())
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)
        saved = repr(f.saver.storage) + repr(f.saver.writes) + repr(f.saver.blobs)
        self.assertIn("The total is 3.", saved)
        self.assertNotIn("You carry out one step", saved)


class AnswerTests(AgentCase):
    async def test_the_typed_answer_reaches_conditions_and_the_interpretation(self):
        echo = make("echo", when="$interpret.ok")
        skipped = make("unless", when="not $interpret.ok")
        f = self.fixture(agent_spec(echo, skipped, outputs={
            "data": ["$total", {"from": "$echo", "optional": True}, {"from": "$unless", "optional": True}],
            "interpretation": "$interpret.text",
        }))
        result = await self.start(f)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.interpretation, "The total is 3.")
        self.assertEqual([call["label"] for _, call in self.script.calls], ["base", "total", "echo"])
        self.assertEqual(await f.entries(result.trace_id), [
            ("use", "ops.make", "base", "done"),
            ("use", "ops.make", "total", "done"),
            ("agent", None, "interpret", "done"),
            ("use", "ops.make", "echo", "done"),
            ("use", "ops.make", "unless", "skipped"),
        ])
        self.assertEqual(len(result.handles), 2)
        steps = {s.step: s for s in await f.runner.steps(await f.env(result.trace_id))}
        self.assertEqual((steps["interpret"].kind, steps["interpret"].result), ("agent", {"text": "The total is 3.", "ok": True}))

    async def test_a_method_on_an_agent_step_is_filled_from_the_run(self):
        f = self.fixture(agent_spec(agent=interpret(method="A model read the total of {total.summary.value}.")))
        result = await self.start(f)
        steps = {s.step: s for s in await f.runner.steps(await f.env(result.trace_id))}
        self.assertEqual(steps["interpret"].method, "A model read the total of 3.")

    async def test_an_agent_step_whose_condition_is_false_is_skipped(self):
        f = self.fixture(agent_spec(
            agent=interpret(when='$quantity == "nothing"'),
            outputs={"data": ["$total"], "interpretation": {"from": "$interpret.text", "optional": True}},
        ))
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)
        self.assertIsNone(result.interpretation)
        self.assertEqual((await f.entries(result.trace_id))[-1], ("agent", None, "interpret", "skipped"))
        self.assertEqual(self.model.calls, 0)


class ShapeTests(AgentCase):
    async def test_a_wrong_shaped_answer_is_retried_once(self):
        for name, wrong in (
            ("a missing field", tool_call(ANSWER_TOOL, {"text": "No ok."})),
            ("a wrong type", tool_call(ANSWER_TOOL, {"text": 3, "ok": True})),
            ("an extra field", tool_call(ANSWER_TOOL, {"text": "x", "ok": True, "more": 1})),
            ("text too long", tool_call(ANSWER_TOOL, {"text": "x" * (MAX_TEXT_CHARS + 1), "ok": True})),
            ("plain text", answer("The total is 3.")),
        ):
            with self.subTest(name):
                self.model = ScriptedChatModel(script=[wrong, said()])
                f = self.fixture(agent_spec())
                result = await self.start(f)
                self.assertEqual(result.status, "done", result.reason)
                self.assertEqual(result.interpretation, "The total is 3.")
                self.assertEqual(self.model.calls, 2)
                retry = self.model.received[1]
                self.assertIs(type(retry[-1]), HumanMessage if name == "plain text" else ToolMessage)
                self.assertEqual(retry[-1].content, CORRECTION)
                entry = await f.entry(result.trace_id, "interpret")
                self.assertEqual(entry.result["model_calls"], 2)

    async def test_two_wrong_shaped_answers_fail_the_step(self):
        for name, script in (
            ("wrong twice", [tool_call(ANSWER_TOOL, {"text": 1}), tool_call(ANSWER_TOOL, {"ok": "yes"})]),
            ("plain text twice", [answer("no"), answer("still no")]),
        ):
            with self.subTest(name):
                self.model = ScriptedChatModel(script=[*script, said()])
                f = self.fixture(agent_spec())
                result = await self.start(f)
                category, wording = MODEL_ANSWER_INVALID
                self.assertEqual((result.status, result.reason), ("failed", wording.format(name="step interpret")))
                self.assertEqual(self.model.calls, 2)
                entry = await f.entry(result.trace_id, "interpret")
                self.assertEqual((entry.status, entry.failure_category), ("failed", category))
                self.assertEqual(entry.result["model_calls"], 2)
                self.assertNotIn("answer", entry.result)


class ModelFailureTests(AgentCase):
    async def test_a_failing_model_falls_back(self):
        primary = ScriptedChatModel(script=[rate_limited(), rate_limited(), rate_limited()])
        fallback = ScriptedChatModel(script=[said(text="From the fallback.")])
        models = ScriptedModels({"agent_step": choice((primary, "fake:primary"), (fallback, "fake:fallback"))})
        f = self.fixture(agent_spec(), models=models)
        result = await self.start(f)

        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.interpretation, "From the fallback.")
        self.assertEqual((primary.calls, fallback.calls), (3, 1))
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual((entry.result["model"], entry.result["model_calls"], entry.result["failed_model_calls"]),
                         ("fake:fallback", 1, 3))
        self.assertEqual(models.asked, ["agent_step"])

    async def test_every_model_failing_fails_the_step_without_its_error_text(self):
        primary = ScriptedChatModel(script=[RuntimeError("primary down " + SECRET)])
        fallback = ScriptedChatModel(script=[rate_limited(), rate_limited(), RuntimeError("fallback down " + SECRET)])
        f = self.fixture(agent_spec(), models=ScriptedModels({"agent_step": choice((primary, "fake:primary"), (fallback, "fake:fallback"))}))
        result = await self.start(f)

        category, wording = MODEL_FAILED
        self.assertEqual((result.status, result.reason), ("failed", wording.format(name="step interpret")))
        self.assertEqual((primary.calls, fallback.calls), (1, 3))
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual((entry.status, entry.failure_category, entry.reason), ("failed", category, result.reason))
        self.assertEqual(entry.result, {"model": None, "model_calls": 0, "failed_model_calls": 4,
                                        "input_tokens": 0, "output_tokens": 0})
        recorded = json.dumps([dataclasses.asdict(e) for e in await f.stores.traces.entries("user-a", result.trace_id)],
                              default=str)
        for text in (recorded, result.model_dump_json()):
            self.assertNotIn(SECRET, text)
            self.assertNotIn("down", text)

    async def test_a_run_without_a_model_for_agent_steps_fails_the_step_with_that_reason(self):
        for name, models in (("no models", None), ("none for the role", ScriptedModels({"explore": choice((self.model, "fake:x"))}))):
            with self.subTest(name):
                f = self.fixture(agent_spec(), models=models)
                result = await self.start(f)
                self.assertEqual((result.status, result.reason), ("failed", NO_MODEL_REASON.format(role="agent_step")))
                entry = await f.entry(result.trace_id, "interpret")
                self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.INTERNAL_ERROR))
        self.assertEqual(self.model.calls, 0)


    async def test_a_failure_outside_the_model_calls_is_an_internal_error(self):
        def broken(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("middleware bug " + SECRET)

        f = self.fixture(agent_spec())
        with patch.object(agent_module._OneRetry, "after_model", broken):
            result = await self.start(f)
        self.assertEqual((result.status, result.reason), ("failed", AGENT_INTERNAL.format(step="interpret")))
        self.assertEqual(self.model.calls, 1)
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.INTERNAL_ERROR))
        self.assertEqual((entry.result["model_calls"], entry.result["failed_model_calls"]), (1, 0))
        recorded = json.dumps(dataclasses.asdict(entry), default=str)
        for text in (recorded, result.model_dump_json()):
            self.assertNotIn(SECRET, text)


class RequiredTests(AgentCase):
    async def test_a_required_agent_step_that_fails_fails_the_run(self):
        self.model = ScriptedChatModel(script=[answer("no"), answer("no")])
        f = self.fixture(agent_spec(make("after", when="$interpret.ok")))
        result = await self.start(f)
        self.assertEqual(result.status, "failed")
        self.assertEqual(await f.run_status(result.trace_id), "failed")
        self.assertEqual([e[2] for e in await f.entries(result.trace_id)], ["base", "total", "interpret"])
        self.assertIsNone(result.interpretation)
        # A failed run keeps what it finished.
        self.assertEqual(len(result.handles), 1)

    async def test_an_optional_agent_step_that_fails_is_a_gap(self):
        self.model = ScriptedChatModel(script=[answer("no"), answer("no")])
        f = self.fixture(agent_spec(
            make("after", when="$interpret.ok"),
            agent=interpret(required=False),
            outputs={"data": ["$total", {"from": "$after", "optional": True}],
                     "interpretation": {"from": "$interpret.text", "optional": True}},
        ))
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.gaps, (Gap(step="interpret", reason=MODEL_ANSWER_INVALID[1].format(name="step interpret")),))
        self.assertIsNone(result.interpretation)
        self.assertEqual((await f.entries(result.trace_id))[2:], [
            ("agent", None, "interpret", "failed"), ("use", "ops.make", "after", "skipped"),
        ])


class TraceEntryTests(AgentCase):
    async def test_the_entry_records_the_references_the_notes_the_answer_and_the_usage(self):
        f = self.fixture(agent_spec())
        result = await self.start(f)
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual((entry.kind, entry.capability, entry.status), ("agent", None, "done"))
        self.assertEqual(entry.params, {
            "reads": ["$total.summary", "$base.product", "$quantity"],
            "notes": ["column-not-surface", "tempo-hours"],
        })
        self.assertEqual(entry.result, {
            "answer": {"text": "The total is 3.", "ok": True},
            "model": "fake:primary", "model_calls": 1, "failed_model_calls": 0,
            "input_tokens": 12, "output_tokens": 7,
        })

    async def test_an_answer_too_large_for_the_trace_is_recorded_as_a_marker(self):
        returns = {"a": "string", "b": "string", "c": "string"}
        long = {name: name * MAX_TEXT_CHARS for name in returns}
        self.assertGreater(len(json.dumps(long)), MAX_JSON_BYTES)
        self.model = ScriptedChatModel(script=[tool_call(ANSWER_TOOL, long)])
        f = self.fixture(agent_spec(
            agent=interpret(agent={"returns": returns}),
            outputs={"data": ["$total"], "interpretation": "$interpret.a"},
        ))
        result = await self.start(f)
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.interpretation, long["a"])
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual(entry.result["answer"], ANSWER_NOT_RECORDED)
        self.assertEqual(entry.result["model"], "fake:primary")


def dates(n: int) -> list[str]:
    first = datetime.date(2024, 1, 1)
    return [(first + datetime.timedelta(days=i)).isoformat() for i in range(n)]


class ReadBoundTests(AgentCase):
    def bounded(self, names: list[str]) -> dict[str, Any]:
        return spec(
            [make("base"), {"id": "interpret", "agent": {
                "instruction": INSTRUCTION, "reads": [f"${n}" for n in names], "returns": {"text": "string"}}}],
            {"data": ["$base"], "interpretation": "$interpret.text"},
            inputs={n: {"type": "date_list", "ask": "Which days?"} for n in names},
        )

    async def test_one_value_past_the_bound_fails_the_step_before_the_model_is_asked(self):
        f = self.fixture(self.bounded(["days"]))
        self.assertGreater(len(json.dumps(dates(300))), MAX_READ_BYTES)
        result = await f.start({"days": dates(300)})
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, f"Step interpret reads $days, which is larger than an agent step may read ({MAX_READ_BYTES} bytes).")
        entry = await f.entry(result.trace_id, "interpret")
        self.assertEqual((entry.kind, entry.status, entry.params["reads"]), ("agent", "failed", ["$days"]))
        self.assertEqual(entry.failure_category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(self.model.calls, 0)

    async def test_values_past_the_bound_together_fail_the_step(self):
        names = [f"days_{i}" for i in range(5)]
        f = self.fixture(self.bounded(names))
        result = await f.start({n: dates(250) for n in names})
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, (
            f"The values step interpret reads are larger together than an agent step may read ({MAX_READS_BYTES} bytes)."
        ))
        self.assertEqual(self.model.calls, 0)

    async def test_values_within_the_bound_are_read(self):
        self.model = ScriptedChatModel(script=[tool_call(ANSWER_TOOL, {"text": "Read."})])
        f = self.fixture(self.bounded(["days"]))
        result = await f.start({"days": dates(250)})
        self.assertEqual(result.status, "done", result.reason)
        self.assertEqual(result.interpretation, "Read.")


if __name__ == "__main__":
    unittest.main()
