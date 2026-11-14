"""The capability framework on in-memory fakes: declaration, invoke path, results, registry."""
import datetime
import json
import tempfile
import unittest
from typing import Annotated, Any

import pandas as pd
from pydantic import Field, ValidationError

from tta_backend.harness.capabilities import (
    Call,
    Capability,
    CapabilityRegistry,
    CapabilityResult,
    FieldNotFound,
    Group,
    InputEvicted,
    Params,
    Question,
    UnknownCapability,
    UnknownGroup,
    begin,
    complete,
    handle_param,
    invoke,
    needs_choice,
)
from tta_backend.harness.capabilities.failures import NO_CONNECTOR_CATEGORIES, SOURCE_CATEGORIES
from tta_backend.harness.handles import Descriptor, HandleError, HandleRecord, ScalarInput, ServicePayload
from tta_backend.harness.run_context import Limits, RunEnv
from tta_backend.harness.storage import memory_stores
from tta_backend.harness.trace import MAX_JSON_BYTES, FailureCategory, StepFailure
from tta_backend.harness.testing import (
    FakeEarthdata,
    FakeToolError,
    RecordingEvents,
    StaticCredentials,
    make_run_context,
    make_run_env,
)

NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
SECRET = "s3cret-token-value"


# Fake capabilities, also used by the Postgres test.

class ScaleParams(Params):
    data: Annotated[str, handle_param("scalar")]
    factor: float = Field(gt=0)


async def _scale(call: Call[ScaleParams]) -> CapabilityResult:
    value = (await call.open("data")).value
    scaled = await call.compute(lambda v, f: v * f, value, call.params.factor)
    record = await call.mint("scalar", ScalarInput(value=scaled), Descriptor(summary="Scaled value", unit="ppb"))
    return call.done(record, summary={"value": scaled}, values={"factor_used": call.params.factor})


scale = Capability(
    name="operations.scale",
    description="Multiply a scalar by a factor.",
    params=ScaleParams,
    output_kind="scalar",
    label="Scaling",
    method="The value was multiplied by {factor}, giving {summary.value} {unit}.",
    retrieves=False,
    retry_safe=True,
    run=_scale,
)


class SumParams(Params):
    items: Annotated[list[str], handle_param("scalar")]
    extra: Annotated[str | None, handle_param("scalar")] = None


async def _sum(call: Call[SumParams]) -> CapabilityResult:
    values = [scalar.value for scalar in await call.open_each("items")]
    record = await call.mint("scalar", ScalarInput(value=sum(values)), Descriptor(summary="Sum"))
    return call.done(record)


total = Capability(
    name="operations.sum", description="Add scalars.", params=SumParams, output_kind="scalar",
    label="Adding", method="The values were added.", retrieves=False, retry_safe=True, run=_sum,
)


class GetParams(Params):
    variable: str


async def _submit(call: Call[GetParams]) -> dict[str, Any]:
    reply = await call.earthdata.call("submit", {"variable": call.params.variable})
    return {"job_id": reply["job_id"]}


async def _finish(call: Call[GetParams], job: dict[str, Any]) -> CapabilityResult:
    reply = await call.earthdata.call("result", {"job_id": job["job_id"]})
    record = await call.record_retrieval(
        reply["handle"], "grid", Descriptor(summary="NO2 grid", unit="molec/cm2", product="TEMPO_NO2_L3")
    )
    return call.done(record, quality={"valid_fraction": 0.8})


get_data = Capability(
    name="sources.fake.get_data",
    description="Retrieve a gridded variable.",
    params=GetParams,
    output_kind="grid",
    label="Retrieving data",
    method="{product} was retrieved for {variable}.",
    retrieves=True,
    retry_safe=True,
    connector="earthdata",
    begin=_submit,
    complete=_finish,
)


def scripted_earthdata() -> FakeEarthdata:
    return FakeEarthdata({
        "submit": lambda args: {"job_id": "job_1"},
        "result": lambda args: {"handle": "obs_abc"},
    })


class RowsParams(Params):
    tables: Annotated[list[str], handle_param("table")]


async def _rows(call: Call[RowsParams]) -> CapabilityResult:
    frames = await call.open_each("tables")
    return call.done(values={"rows": sum(len(f) for f in frames)})


count_rows = Capability(
    name="operations.count_rows", description="Count table rows.", params=RowsParams, output_kind=None,
    label="Counting", method="Rows were counted.", retrieves=False, retry_safe=True, run=_rows,
)


def two_phase(name: str, submit: Any, finish: Any = _finish) -> Capability[GetParams]:
    return Capability(
        name=name, description="A test retrieval.", params=GetParams, output_kind="grid",
        label="Retrieving", method="Retrieved.", retrieves=True, retry_safe=True, connector="earthdata",
        begin=submit, complete=finish,
    )


class NoParams(Params):
    pass


def outcome_capability(name: str, outcome: Any, output_kind: Any = None) -> Capability[NoParams]:
    """A capability whose run returns ``outcome``, or raises it when it is an exception."""

    async def run(call: Call[NoParams]) -> CapabilityResult:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[no-any-return]

    return Capability(
        name=name, description="A test capability.", params=NoParams, output_kind=output_kind,
        label="Testing", method="Nothing was computed.", retrieves=False, retry_safe=True, run=run,
    )


async def seed_scalar(env: RunEnv, value: float, user_id: str | None = None) -> HandleRecord:
    return await env.resolver.mint(
        user_id=user_id or env.context.identity.user_id, conversation_id="conv-1", kind="scalar",
        data=ScalarInput(value=value), descriptor=Descriptor(summary="Seed"), operation="test.seed",
        created_at=NOW,
    )


async def seed_table(env: RunEnv, rows: int) -> HandleRecord:
    return await env.resolver.mint(
        user_id=env.context.identity.user_id, conversation_id="conv-1", kind="table",
        data=pd.DataFrame({"a": range(rows)}), descriptor=Descriptor(summary="Rows"),
        operation="test.seed", created_at=NOW,
    )


class _EnvCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stores = memory_stores()
        self.events = RecordingEvents()
        self.earthdata = scripted_earthdata()
        self.env = await self.make_env("user-a", credentials=StaticCredentials({"earthdata": SECRET}))

    async def make_env(self, user_id: str, credentials: Any = None) -> RunEnv:
        context = make_run_context(
            user_id=user_id, stores=self.stores, events=self.events, credentials=credentials
        )
        return await make_run_env(self._tmp.name, context=context, earthdata=self.earthdata)

    async def entries(self, env: RunEnv | None = None) -> list[Any]:
        env = env or self.env
        return await self.stores.traces.entries(env.context.identity.user_id, env.run_id)

    async def failure(self, capability: Capability[Any], params: dict[str, Any]) -> StepFailure:
        with self.assertRaises(StepFailure) as caught:
            await invoke(self.env, capability, params)
        return caught.exception


class InvokeTests(_EnvCase):
    async def test_success_mints_with_lineage_and_records_one_use_entry(self):
        seed = await seed_scalar(self.env, 2.0)
        result = await invoke(self.env, scale, {"data": seed.id, "factor": 3}, step="scaled")

        self.assertEqual((result.status, result.kind, result.unit), ("done", "scalar", "ppb"))
        self.assertEqual(result.summary, {"value": 6.0})
        self.assertEqual(result.method, "The value was multiplied by 3, giving 6 ppb.")
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (seed.id,))
        self.assertEqual(record.operation, "operations.scale")
        self.assertEqual(record.run_id, self.env.run_id)
        self.assertEqual(record.parameters, {"data": seed.id, "factor": 3.0})

        [entry] = await self.entries()
        self.assertEqual((entry.kind, entry.capability, entry.step, entry.status), ("use", "operations.scale", "scaled", "done"))
        self.assertEqual(entry.params, {"data": seed.id, "factor": 3.0})
        self.assertEqual(entry.result["handle"], result.handle)
        self.assertEqual(entry.result["method"], result.method)
        self.assertEqual(self.events.progress_events[-1], {"message": "Scaling", "stage": "operations.scale", "detail": None})

    async def test_list_and_optional_handle_params_resolve_and_become_parents(self):
        a, b = await seed_scalar(self.env, 1.0), await seed_scalar(self.env, 2.0)
        result = await invoke(self.env, total, {"items": [a.id, b.id, a.id]})
        record = await self.stores.handles.get("user-a", result.handle)
        self.assertEqual(record.parents, (a.id, b.id))

    async def test_invalid_params_are_refused_before_running_without_echoing_unknown_keys(self):
        seed = await seed_scalar(self.env, 2.0)
        cases = [
            ({"data": seed.id}, "factor"),
            ({"data": seed.id, "factor": -1}, "factor"),
            ({"data": seed.id, "factor": 1, "api_key_zzz": "x"}, "an unexpected parameter"),
        ]
        for params, named in cases:
            with self.subTest(params=params):
                failure = await self.failure(scale, params)
                self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
                self.assertIn(named, failure.reason)
                self.assertNotIn("api_key_zzz", failure.reason)
        entries = await self.entries()
        self.assertEqual([e.status for e in entries], ["failed"] * 3)
        self.assertEqual([e.params for e in entries], [
            {"data": seed.id}, {"data": seed.id, "factor": -1}, {"data": seed.id, "factor": 1},
        ])
        self.assertNotIn("api_key_zzz", json.dumps([e.params for e in entries]))
        self.assertEqual(self.events.progress_events, [])

    async def test_invalid_declared_params_too_large_to_record_are_left_out(self):
        await self.failure(scale, {"data": "x" * 5000, "factor": 1, "token": "y"})
        [entry] = await self.entries()
        self.assertEqual(entry.params, {})

    async def test_params_too_large_to_record_are_refused(self):
        failure = await self.failure(get_data, {"variable": "x" * 5000})
        self.assertEqual(failure.category, FailureCategory.BAD_INPUT)
        [entry] = await self.entries()
        self.assertEqual(entry.params, {})

    async def test_another_users_handle_is_refused_exactly_as_a_missing_one(self):
        other = await self.make_env("user-b")
        theirs = await seed_scalar(other, 5.0)
        foreign = await self.failure(scale, {"data": theirs.id, "factor": 1})
        missing = await self.failure(scale, {"data": "scalar_" + "0" * 32, "factor": 1})
        self.assertEqual(foreign.category, FailureCategory.BAD_INPUT)
        self.assertEqual(foreign.reason.replace(theirs.id, ""), missing.reason)
        self.assertEqual(foreign.reason, "operations.scale: there is no handle for data.")

    async def test_wrong_kind_is_refused_with_the_accepted_kinds(self):
        grid = await self.stores.handles.record_retrieval(HandleRecord(
            id="obs_grid", kind="grid", origin="retrieval", user_id="user-a", conversation_id="conv-1",
            descriptor=Descriptor(summary="A grid"), created_at=NOW, payload=ServicePayload(),
        ))
        failure = await self.failure(scale, {"data": grid.id, "factor": 1})
        self.assertEqual(failure.reason, "operations.scale: data must be a handle of kind scalar.")

    async def test_missing_connector_fails_before_anything_runs(self):
        env = await self.make_env("user-c", credentials=StaticCredentials())
        with self.assertRaises(StepFailure) as caught:
            await invoke(env, get_data, {"variable": "no2"})
        self.assertEqual(caught.exception.category, FailureCategory.CREDENTIAL_INVALID)
        self.assertIn("earthdata", caught.exception.reason)
        self.assertEqual(self.earthdata.calls, [])
        [entry] = await self.entries(env)
        self.assertEqual(entry.failure_category, FailureCategory.CREDENTIAL_INVALID)

    async def test_the_secret_reaches_the_call_but_nothing_durable(self):
        seen = []

        async def run(call: Call[NoParams]) -> CapabilityResult:
            seen.append(call.secret)
            return call.done()

        uses_secret = Capability(
            name="sources.fake.lookup", description="Look something up.", params=NoParams, output_kind=None,
            label="Looking up", method="A lookup ran.", retrieves=False, retry_safe=True, connector="earthdata", run=run,
        )
        result = await invoke(self.env, uses_secret, {})
        await invoke(self.env, get_data, {"variable": "no2"})
        self.assertEqual(seen, [SECRET])
        entries = await self.entries()
        durable = json.dumps([e.params for e in entries] + [e.result for e in entries] + [result.recorded()])
        durable += (await self.stores.handles.get("user-a", "obs_abc")).model_dump_json()
        self.assertNotIn(SECRET, durable)

    async def test_needs_input_is_recorded_as_waiting(self):
        question = Question(kind="pick_one", prompt="Which variable?", options=("no2", "hcho"))
        asks = outcome_capability("operations.ask", CapabilityResult(status="needs_input", question=question))
        result = await invoke(self.env, asks, {})
        self.assertEqual(result.question, question)
        self.assertIsNone(result.method)
        [entry] = await self.entries()
        self.assertEqual(entry.status, "waiting")
        self.assertEqual(entry.result["question"]["options"], ["no2", "hcho"])

    async def test_insufficient_evidence_is_returned_and_recorded_as_no_data(self):
        stop = CapabilityResult(
            status="insufficient_evidence", reason="Only 2 of 5 days had valid data.",
            values={"nearest_dates": ["2026-01-02"]},
        )
        result = await invoke(self.env, outcome_capability("operations.short", stop), {})
        self.assertEqual(result.read("nearest_dates"), ["2026-01-02"])
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category, entry.reason),
                         ("failed", FailureCategory.NO_DATA, "Only 2 of 5 days had valid data."))
        self.assertEqual(entry.result, result.recorded())
        self.assertEqual(entry.result["values"], {"nearest_dates": ["2026-01-02"]})

    async def test_an_evicted_input_names_the_handles_to_rebuild(self):
        kept, gone_a, gone_b = [await seed_table(self.env, rows) for rows in (1, 2, 3)]
        for record in (gone_a, gone_b):
            await self.env.resolver.evict("user-a", record.id, NOW)
        failure = await self.failure(count_rows, {"tables": [gone_a.id, kept.id, gone_b.id]})
        self.assertIsInstance(failure, InputEvicted)
        self.assertEqual(failure.category, FailureCategory.INPUT_EVICTED)
        self.assertEqual(failure.handle_ids, (gone_a.id, gone_b.id))
        [entry] = await self.entries()
        self.assertEqual(entry.failure_category, FailureCategory.INPUT_EVICTED)
        self.assertEqual(entry.result, {"evicted": [gone_a.id, gone_b.id]})

    async def test_a_secret_in_capability_output_is_refused_without_echoing_it(self):
        async def leaks_result(call: Call[NoParams]) -> CapabilityResult:
            return call.done(values={"url": f"https://host/x?token={call.secret}"})

        async def leaks_descriptor(call: Call[NoParams]) -> CapabilityResult:
            record = await call.mint("scalar", ScalarInput(value=1), Descriptor(summary=f"key {call.secret}"))
            return call.done(record)

        async def leaks_parameters(call: Call[NoParams]) -> CapabilityResult:
            record = await call.mint("scalar", ScalarInput(value=1), Descriptor(summary="ok"),
                                     parameters={"auth": call.secret})
            return call.done(record)

        async def leaks_job(call: Call[GetParams]) -> dict[str, Any]:
            return {"job_id": "job_1", "auth": call.secret}

        def connected(name: str, run: Any, kind: Any = None) -> Capability[NoParams]:
            return Capability(
                name=name, description="Leak.", params=NoParams, output_kind=kind, label="Leaking",
                method="Leaked.", retrieves=False, retry_safe=True, connector="earthdata", run=run,
            )

        cases = [
            (connected("sources.fake.result", leaks_result), {}),
            (connected("sources.fake.descriptor", leaks_descriptor, "scalar"), {}),
            (connected("sources.fake.parameters", leaks_parameters, "scalar"), {}),
            (two_phase("sources.fake.job", leaks_job), {"variable": "no2"}),
        ]
        for capability, params in cases:
            with self.subTest(capability.name):
                failure = await self.failure(capability, params)
                self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
                self.assertIn("connector secret", failure.reason)
                self.assertNotIn(SECRET, failure.reason)
        entries = await self.entries()
        self.assertEqual([e.status for e in entries], ["failed"] * len(cases))
        durable = json.dumps([[e.params, e.result, e.reason] for e in entries])
        self.assertNotIn(SECRET, durable)

    async def test_needs_choice_builds_a_pick_question_from_options(self):
        async def run(call: Call[NoParams]) -> CapabilityResult:
            return call.ask("Which variable?", ["no2", "hcho"])

        asks = Capability(
            name="operations.choose", description="Ask.", params=NoParams, output_kind=None, label="Asking",
            method="Asked.", retrieves=False, retry_safe=True, run=run,
        )
        result = await invoke(self.env, asks, {})
        self.assertEqual(result.question, Question(kind="pick_one", prompt="Which variable?", options=("no2", "hcho")))
        many = needs_choice("Which levels?", (1, 2), many=True)
        assert many.question is not None
        self.assertEqual((many.status, many.question.kind, many.question.options), ("needs_input", "pick_many", (1, 2)))

    async def test_a_result_that_breaks_the_declaration_is_an_internal_error(self):
        seed = await seed_scalar(self.env, 1.0)
        foreign = await seed_scalar(await self.make_env("user-b"), 1.0)
        cases = {
            "no handle for a declared output": (CapabilityResult(status="done"), "scalar"),
            "a handle where none is declared": (CapabilityResult(status="done", handle=seed.id, kind="scalar"), None),
            "the wrong output kind": (CapabilityResult(status="done", handle="obs_x", kind="grid"), "scalar"),
            "another user's handle": (CapabilityResult(status="done", handle=foreign.id, kind="scalar"), "scalar"),
            "not a result": ({"status": "done"}, None),
        }
        for label, (outcome, kind) in cases.items():
            with self.subTest(label):
                failure = await self.failure(outcome_capability("operations.bad", outcome, kind), {})
                self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
                self.assertIn("does not match its declaration", failure.reason)


class FailureMappingTests(_EnvCase):
    LEAK = "https://user:pw@example.test/data?token=abc"

    async def test_every_source_category_maps_to_fixed_wording(self):
        expected = {
            "user_input": FailureCategory.BAD_INPUT,
            "no_data": FailureCategory.NO_DATA,
            "not_found": FailureCategory.BAD_INPUT,
            "too_large": FailureCategory.REQUEST_TOO_LARGE,
            "provider_unavailable": FailureCategory.SOURCE_UNAVAILABLE,
            "contract": FailureCategory.INTERNAL_ERROR,
            "unsupported_grid": FailureCategory.UNSUPPORTED_SHAPE,
            "variable_choice_required": FailureCategory.BAD_INPUT,
            "dimension_choice_required": FailureCategory.BAD_INPUT,
            "token_invalid": FailureCategory.CREDENTIAL_INVALID,
            "token_expired": FailureCategory.CREDENTIAL_INVALID,
            "eula_not_accepted": FailureCategory.CREDENTIAL_INVALID,
            "something_new": FailureCategory.INTERNAL_ERROR,
        }
        self.assertEqual(set(SOURCE_CATEGORIES) | {"something_new"}, set(expected))
        for category, mapped in expected.items():
            with self.subTest(category):
                raises = outcome_capability("sources.fake.raises", FakeToolError(category, self.LEAK))
                failure = await self.failure(raises, {})
                self.assertEqual(failure.category, mapped)
                self.assertNotIn("example.test", failure.reason)
                rules = {**SOURCE_CATEGORIES, **NO_CONNECTOR_CATEGORIES}
                wording = rules.get(category, (None, "{name} hit an internal error."))[1]
                self.assertEqual(failure.reason, wording.format(name="sources.fake.raises"))
                self.assertNotIn("source credential", failure.reason)
                self.assertNotIn("with source", failure.reason)
        entries = await self.entries()
        self.assertEqual(len(entries), len(expected))
        self.assertTrue(all(e.status == "failed" and "example.test" not in (e.reason or "") for e in entries))

    async def test_plain_exceptions_are_internal_errors_with_fixed_wording(self):
        failure = await self.failure(outcome_capability("operations.crash", RuntimeError(self.LEAK)), {})
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(failure.reason, "operations.crash hit an internal error.")
        self.assertIsNone(failure.__cause__)
        self.assertIsNone(failure.__context__)

    async def test_internal_errors_are_logged_once_without_the_message(self):
        with self.assertLogs("tta_backend.harness.capabilities", level="ERROR") as logs:
            await self.failure(outcome_capability("operations.crash", RuntimeError(self.LEAK)), {})
        [line] = logs.output
        self.assertIn("operations.crash", line)
        self.assertIn(self.env.run_id, line)
        self.assertIn("RuntimeError", line)
        self.assertNotIn("example.test", line)
        self.assertEqual([r.exc_info for r in logs.records], [None])

    async def test_bad_input_is_not_logged_as_an_error(self):
        with self.assertNoLogs("tta_backend.harness.capabilities", level="ERROR"):
            await self.failure(outcome_capability("sources.fake.raises", FakeToolError("user_input")), {})

    async def test_handle_errors_raised_while_running_map(self):
        cases = (
            (HandleError("evicted", "table_" + "a" * 32), FailureCategory.INPUT_EVICTED),
            (HandleError("not_found"), FailureCategory.BAD_INPUT),
            (HandleError("already_exists"), FailureCategory.INTERNAL_ERROR),
        )
        for error, category in cases:
            with self.subTest(error.reason):
                failure = await self.failure(outcome_capability("operations.handles", error), {})
                self.assertEqual(failure.category, category)
        evicted = (await self.entries())[0]
        self.assertEqual(evicted.result, {"evicted": ["table_" + "a" * 32]})

    async def test_a_capabilitys_own_step_failure_passes_through(self):
        own = StepFailure(FailureCategory.NO_DATA, "No monitor reported on that day.")
        failure = await self.failure(outcome_capability("operations.own", own), {})
        self.assertIs(failure, own)


class TwoPhaseTests(_EnvCase):
    async def test_begin_then_complete_submits_once_and_records_both_phases(self):
        job = await begin(self.env, get_data, {"variable": "no2"}, step="event")
        self.assertEqual(job, {"job_id": "job_1"})
        result = await complete(self.env, get_data, {"variable": "no2"}, job, step="event")

        self.assertEqual((result.handle, result.kind, result.product), ("obs_abc", "grid", "TEMPO_NO2_L3"))
        self.assertEqual(result.method, "TEMPO_NO2_L3 was retrieved for no2.")
        self.assertEqual([tool for tool, _ in self.earthdata.calls], ["submit", "result"])
        entries = await self.entries()
        self.assertEqual([(e.step, e.status) for e in entries], [("event", "waiting"), ("event", "done")])
        self.assertEqual(entries[0].result, {"job": {"job_id": "job_1"}})
        self.assertEqual(entries[0].params, {"variable": "no2"})
        self.assertEqual(entries[1].params, {"variable": "no2", "_job": {"job_id": "job_1"}})

    async def test_complete_after_a_restart_never_submits(self):
        result = await complete(self.env, get_data, {"variable": "no2"}, {"job_id": "job_1"})
        self.assertEqual(result.handle, "obs_abc")
        self.assertEqual([tool for tool, _ in self.earthdata.calls], ["result"])

    async def test_plain_invoke_saves_the_job_before_completing(self):
        result = await invoke(self.env, get_data, {"variable": "no2"}, step="event")
        self.assertEqual(result.handle, "obs_abc")
        self.assertEqual([tool for tool, _ in self.earthdata.calls], ["submit", "result"])
        entries = await self.entries()
        self.assertEqual([(e.step, e.status) for e in entries], [("event", "waiting"), ("event", "done")])
        self.assertEqual(entries[0].result, {"job": {"job_id": "job_1"}})
        self.assertEqual(entries[1].params["_job"], {"job_id": "job_1"})

    async def test_a_failed_complete_leaves_the_job_saved(self):
        async def fails(call: Call[GetParams], job: dict[str, Any]) -> CapabilityResult:
            raise FakeToolError("provider_unavailable", "https://host/job?token=abc")

        flaky = two_phase("sources.fake.flaky", _submit, fails)
        failure = await self.failure(flaky, {"variable": "no2"})
        self.assertEqual(failure.category, FailureCategory.SOURCE_UNAVAILABLE)
        self.assertEqual([tool for tool, _ in self.earthdata.calls], ["submit"])
        waiting, failed = await self.entries()
        self.assertEqual((waiting.status, waiting.result), ("waiting", {"job": {"job_id": "job_1"}}))
        self.assertEqual((failed.status, failed.params["_job"]), ("failed", {"job_id": "job_1"}))

    async def test_a_job_that_fits_alone_but_not_in_its_entry_is_a_recorded_step_failure(self):
        # The job alone is exactly at the limit; wrapped as {"job": ...} it is over.
        blob = "x" * (MAX_JSON_BYTES - len(json.dumps({"blob": ""}, separators=(",", ":"))))
        self.assertEqual(len(json.dumps({"blob": blob}, separators=(",", ":"))), MAX_JSON_BYTES)

        async def near_limit(call: Call[GetParams]) -> dict[str, Any]:
            await call.earthdata.call("submit", {})
            return {"blob": blob}

        failure = await self.failure(two_phase("sources.fake.near", near_limit), {"variable": "no2"})
        self.assertEqual(failure.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual([tool for tool, _ in self.earthdata.calls], ["submit"])
        [entry] = await self.entries()
        self.assertEqual((entry.status, entry.failure_category), ("failed", FailureCategory.INTERNAL_ERROR))

    async def test_a_complete_given_an_unrecordable_job_is_refused_before_running(self):
        with self.assertRaises(StepFailure) as caught:
            await complete(self.env, get_data, {"variable": "no2"}, {"blob": "x" * 5000})
        self.assertEqual(caught.exception.category, FailureCategory.INTERNAL_ERROR)
        self.assertEqual(self.earthdata.calls, [])
        [entry] = await self.entries()
        self.assertEqual(entry.params, {})

    async def test_the_idempotency_key_is_stable_per_run_and_step(self):
        keys: list[str] = []

        async def submit(call: Call[GetParams]) -> dict[str, Any]:
            keys.append(call.idempotency_key)
            return {"job_id": "job_1"}

        keyed = two_phase("sources.fake.keyed", submit)
        other = await self.make_env("user-a", credentials=StaticCredentials({"earthdata": SECRET}))
        calls = ((self.env, "event"), (self.env, "event"), (other, "event"), (self.env, "baseline"), (self.env, None))
        for env, step in calls:
            await begin(env, keyed, {"variable": "no2"}, step=step)
        first, again, other_run, other_step, unnamed = keys
        self.assertEqual(first, again)
        self.assertEqual(len({first, other_run, other_step, unnamed}), 4)
        self.assertRegex(first, r"^tta-[0-9a-f]{32}$")

    async def test_single_phase_capabilities_cannot_be_split(self):
        with self.assertRaises(TypeError):
            await begin(self.env, scale, {})
        with self.assertRaises(TypeError):
            await complete(self.env, scale, {}, {})

    async def test_a_job_reference_must_be_small_json(self):
        async def huge(call: Call[GetParams]) -> dict[str, Any]:
            return {"blob": "x" * 5000}

        big = Capability(
            name="sources.fake.big", description="Submit a big job.", params=GetParams, output_kind="grid",
            label="Submitting", method="Submitted.", retrieves=True, retry_safe=True, begin=huge, complete=_finish,
        )
        with self.assertRaises(StepFailure) as caught:
            await begin(self.env, big, {"variable": "no2"})
        self.assertEqual(caught.exception.category, FailureCategory.INTERNAL_ERROR)


class ResultTests(unittest.TestCase):
    def test_field_paths_reach_fields_values_and_nested_entries(self):
        result = CapabilityResult(
            status="done", handle="grid_" + "a" * 32, kind="grid", unit="ppb", product="TEMPO_NO2_L3",
            summary={"mean": 1.5, "max_location": [40.1, -74.2]},
            values={"days": ["2026-01-01", "2026-01-08"], "rule": "same weekday"},
        )
        self.assertEqual(result.read("summary.mean"), 1.5)
        self.assertEqual(result.read("summary.max_location.1"), -74.2)
        self.assertEqual(result.read("days"), ["2026-01-01", "2026-01-08"])
        self.assertEqual(result.read("rule"), "same weekday")
        self.assertEqual(result.read("product"), "TEMPO_NO2_L3")
        self.assertEqual(result.read("unit"), "ppb")
        for missing in ("nothing", "summary.median", "days.5", "summary.mean.x"):
            with self.subTest(missing), self.assertRaises(FieldNotFound):
                result.read(missing)

    def test_results_hold_small_values_only(self):
        with self.assertRaises(ValidationError):
            CapabilityResult(status="done", values={"data": list(range(2000))})
        with self.assertRaises(ValidationError):
            CapabilityResult(status="done", summary={"mean": float("nan")})

    def test_result_shape_rules(self):
        bad = [
            {"status": "done", "handle": "grid_" + "a" * 32},
            {"status": "done", "handle": "grid_" + "a" * 32, "kind": "series"},
            {"status": "done", "handle": "../etc", "kind": "grid"},
            {"status": "needs_input"},
            {"status": "insufficient_evidence"},
            {"status": "done", "reason": "why"},
            {"status": "done", "values": {"unit": "x"}},
            {"status": "done", "values": {"Bad Name": 1}},
        ]
        for fields in bad:
            with self.subTest(fields), self.assertRaises(ValidationError):
                CapabilityResult(**fields)

    def test_question_kinds(self):
        Question(kind="confirm_list", prompt="Confirm the days.", items=("2026-01-01",))
        Question(kind="text", prompt="Which monitor?")
        for fields in (
            {"kind": "pick_one", "prompt": "Pick."},
            {"kind": "pick_many", "prompt": "Pick.", "options": ("a",), "items": ("b",)},
            {"kind": "confirm_list", "prompt": "Confirm.", "options": ("a",)},
            {"kind": "text", "prompt": "Say.", "options": ("a",)},
        ):
            with self.subTest(fields), self.assertRaises(ValidationError):
                Question(**fields)


class DefinitionTests(unittest.TestCase):
    def base(self, **over: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "name": "operations.x", "description": "Do x.", "params": NoParams, "output_kind": None,
            "label": "Doing x", "method": "X was done.", "retrieves": False, "retry_safe": True, "run": _sum,
        }
        return {**fields, **over}

    def test_declarations_are_checked(self):
        bad = [
            {"name": "difference"},
            {"name": "Operations.X"},
            {"description": "two\nlines"},
            {"label": ""},
            {"method": "Unclosed {brace"},
            {"method": "Attribute {x.__class__} access"},
            {"begin": _submit},
            {"run": None},
            {"run": None, "begin": _submit},
            {"params": dict},
        ]
        for over in bad:
            with self.subTest(over), self.assertRaises(ValueError):
                Capability(**self.base(**over))

    def test_handle_params_are_found_and_checked_statically(self):
        self.assertEqual(dict(scale.handle_params), {"data": ("scalar",)})
        self.assertEqual(dict(total.handle_params), {"items": ("scalar",), "extra": ("scalar",)})
        self.assertTrue(scale.accepts("data", "scalar"))
        self.assertFalse(scale.accepts("data", "grid"))
        self.assertFalse(scale.accepts("factor", "scalar"))
        self.assertEqual((scale.group, get_data.group), ("operations", "sources.fake"))
        self.assertEqual((scale.two_phase, get_data.two_phase), (False, True))

    def test_a_nested_handle_marker_is_refused(self):
        class Nested(Params):
            data: Annotated[str, handle_param("grid")] | None = None

        with self.assertRaises(ValueError):
            Capability(**self.base(params=Nested))
        with self.assertRaises(ValueError):
            handle_param()

    def test_limits_are_checked(self):
        ok = {"retrieval_soft_bytes": 1, "retrieval_hard_bytes": 2, "max_point_timeseries_days": 1,
              "retrieval_timeout_seconds": 10.0, "poll_min_seconds": 1.0, "poll_max_seconds": 5.0}
        Limits(**ok)
        for over in ({"retrieval_soft_bytes": 3}, {"poll_min_seconds": 6.0}, {"poll_max_seconds": 11.0}):
            with self.subTest(over), self.assertRaises(ValueError):
                Limits(**{**ok, **over})


class RegistryTests(unittest.TestCase):
    def registry(self) -> CapabilityRegistry:
        registry = CapabilityRegistry()
        registry.add_group(Group("operations", "Combine and reduce handles.", notes="Kinds must match."))
        registry.add_group(Group("sources.fake", "A fake gridded source."))
        for capability in (total, scale, get_data):
            registry.register(capability)
        return registry

    def test_lookup_groups_and_index(self):
        registry = self.registry()
        self.assertIs(registry.get("operations.scale"), scale)
        self.assertEqual([c.name for c in registry.in_group("operations")], ["operations.scale", "operations.sum"])
        self.assertEqual(registry.group("operations").notes, "Kinds must match.")
        self.assertEqual(registry.index(), "operations: Combine and reduce handles.\nsources.fake: A fake gridded source.")
        with self.assertRaises(UnknownCapability):
            registry.get("operations.nothing")
        with self.assertRaises(UnknownGroup):
            registry.in_group("render")

    def test_duplicates_and_undeclared_groups_are_refused(self):
        registry = self.registry()
        with self.assertRaises(ValueError):
            registry.register(scale)
        with self.assertRaises(ValueError):
            registry.add_group(Group("operations", "Again."))
        with self.assertRaises(UnknownGroup):
            registry.register(Capability(**DefinitionTests().base(name="render.map")))
        with self.assertRaises(ValueError):
            Group("Bad Group", "x")

    def test_schema_marks_handle_parameters(self):
        schema = self.registry().schema("operations.scale")
        self.assertEqual(schema["properties"]["data"]["x-handle-kinds"], ["scalar"])
        self.assertEqual(schema["properties"]["factor"]["exclusiveMinimum"], 0)
        self.assertEqual(schema["required"], ["data", "factor"])
        self.assertNotIn("x-handle-kinds", schema["properties"]["factor"])


if __name__ == "__main__":
    unittest.main()
