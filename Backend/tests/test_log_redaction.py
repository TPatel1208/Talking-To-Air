"""
tests/test_log_redaction.py
===========================
EPA AQS authenticates with ``email`` and ``key`` query parameters, so every
AQS request URL carries the credential. httpx logs each request URL at INFO,
and EPA echoes the full URL back in its response Header, which the AQS client
logs on failure. configure_logging must keep both values out of every log
line while leaving the rest of the query (param, bdate, edate, bbox,
state/county/site) readable, since that is how a run's queries get debugged.
"""
import io
import json
import logging
import sys
import unittest
from unittest.mock import patch

import httpx

AQS_EMAIL = "researcher@lab.edu"
AQS_KEY = "s3cr3tAqsKey"
AQS_PARAMS = {
    "param": "88101",
    "bdate": "20240601",
    "edate": "20240607",
    "minlat": "40.1",
    "maxlat": "40.9",
    "minlon": "-74.6",
    "maxlon": "-73.8",
    "state": "34",
    "county": "013",
    "site": "0017",
    "email": AQS_EMAIL,
    "key": AQS_KEY,
}


def _settings(log_format="text"):
    from tta_backend.config.settings import Settings

    return Settings(log_level="DEBUG", log_format=log_format)


class _LoggingSandbox(unittest.TestCase):
    """Runs each test against a clean root logger and httpx logger, restored
    afterwards, so configure_logging's handlers and filters never leak into
    the rest of the suite."""

    def setUp(self):
        root = logging.getLogger()
        self._root_state = (list(root.handlers), list(root.filters), root.level)
        self._logger_filters = {
            name: list(logging.getLogger(name).filters) for name in ("httpx", "httpcore", "uvicorn.error")
        }
        for handler in list(root.handlers):
            root.removeHandler(handler)
        self.addCleanup(self._restore)

    def _restore(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        handlers, filters, level = self._root_state
        for handler in handlers:
            root.addHandler(handler)
        root.filters[:] = filters
        root.setLevel(level)
        for name, filters_before in self._logger_filters.items():
            logging.getLogger(name).filters[:] = filters_before

    def _configure_capturing(self, log_format="text"):
        """configure_logging on an empty root, capturing its stderr handler."""
        from tta_backend.utils.logging import configure_logging

        stream = io.StringIO()
        with patch.object(sys, "stderr", stream):
            configure_logging(_settings(log_format))
        return stream

    def assert_credentials_redacted(self, text):
        self.assertNotIn(AQS_KEY, text)
        self.assertNotIn(AQS_EMAIL, text)
        self.assertNotIn("researcher%40lab.edu", text)
        self.assertIn("REDACTED", text)


class HttpxRequestLineTests(_LoggingSandbox):
    async def _aqs_request_through_httpx(self):
        def answer(request):
            return httpx.Response(200, json={"Header": [{"status": "Success"}], "Data": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            await client.get("https://aqs.epa.gov/data/api/dailyData/byBox", params=AQS_PARAMS)

    def test_the_httpx_request_line_for_an_aqs_call_carries_neither_email_nor_key(self):
        import asyncio

        stream = self._configure_capturing()
        asyncio.run(self._aqs_request_through_httpx())

        output = stream.getvalue()
        self.assertIn("HTTP Request: GET https://aqs.epa.gov/data/api/dailyData/byBox", output)
        self.assert_credentials_redacted(output)
        for name, value in AQS_PARAMS.items():
            if name not in ("email", "key"):
                self.assertIn(f"{name}={value}", output)

    def test_redaction_also_holds_when_root_already_had_a_handler(self):
        """Under uvicorn or pytest the root logger can already have handlers
        when configure_logging runs, and it then adds none of its own. Those
        existing handlers must still never print the credential."""
        import asyncio

        from tta_backend.utils.logging import configure_logging

        stream = io.StringIO()
        logging.getLogger().addHandler(logging.StreamHandler(stream))
        configure_logging(_settings())
        asyncio.run(self._aqs_request_through_httpx())

        self.assert_credentials_redacted(stream.getvalue())

    def test_a_handler_added_after_configure_logging_still_gets_redacted_httpx_lines(self):
        import asyncio

        configure_stream = self._configure_capturing()
        late_stream = io.StringIO()
        logging.getLogger().addHandler(logging.StreamHandler(late_stream))
        asyncio.run(self._aqs_request_through_httpx())

        self.assertTrue(configure_stream.getvalue())
        self.assert_credentials_redacted(late_stream.getvalue())


class BackendLogLineTests(_LoggingSandbox):
    def test_the_echoed_epa_header_logged_by_the_backend_is_redacted(self):
        """EPA's Header repeats the full request URL, email and key included,
        and the AQS client logs that header when EPA reports a failure."""
        stream = self._configure_capturing()
        header = {
            "status": "Failed",
            "url": f"https://aqs.epa.gov/data/api/list/states?email={AQS_EMAIL}&key={AQS_KEY}",
            "error": ["Email and/or key are invalid."],
        }

        logging.getLogger("tta_backend.tools.ground_sensor_tools.epa_aqs_tools").warning(
            "aqs_error endpoint=%s header=%r", "list/states", header,
        )

        output = stream.getvalue()
        self.assert_credentials_redacted(output)
        self.assertIn("Email and/or key are invalid.", output)

    def test_a_logged_traceback_carrying_the_url_is_redacted_in_json_format(self):
        stream = self._configure_capturing(log_format="json")
        request = httpx.Request("GET", "https://aqs.epa.gov/data/api/list/states", params=AQS_PARAMS)
        response = httpx.Response(400, request=request)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logging.getLogger("tta_backend.test").exception("aqs_request_failed")

        record = json.loads(stream.getvalue().strip().splitlines()[-1])
        self.assertIn("HTTPStatusError", record["exc_info"])
        self.assert_credentials_redacted(record["exc_info"])

    def test_other_query_parameters_that_merely_end_in_key_are_left_alone(self):
        stream = self._configure_capturing()

        logging.getLogger("tta_backend.test").info(
            "GET https://example.org/x?monkey=banana&param=88101&key=%s", AQS_KEY,
        )

        output = stream.getvalue()
        self.assertIn("monkey=banana", output)
        self.assertIn("param=88101", output)
        self.assertIn("key=REDACTED", output)
        self.assertNotIn(AQS_KEY, output)

    def test_reconfiguring_does_not_stack_filters(self):
        from tta_backend.utils.logging import configure_logging

        self._configure_capturing()
        configure_logging(_settings())
        configure_logging(_settings())

        def redaction_filters(filterer):
            return [f for f in filterer.filters if type(f).__name__ == "RedactCredentialsFilter"]

        self.assertEqual(len(redaction_filters(logging.getLogger("httpx"))), 1)
        for handler in logging.getLogger().handlers:
            self.assertEqual(len(redaction_filters(handler)), 1)


    def test_uvicorns_own_error_logger_redacts_an_app_traceback(self):
        """uvicorn logs an unhandled app exception on uvicorn.error, which has
        its own handler and does not propagate to root."""
        self._configure_capturing()
        uvicorn_stream = io.StringIO()
        uvicorn_logger = logging.getLogger("uvicorn.error")
        handler = logging.StreamHandler(uvicorn_stream)
        uvicorn_logger.addHandler(handler)
        self.addCleanup(uvicorn_logger.removeHandler, handler)
        request = httpx.Request("GET", "https://aqs.epa.gov/data/api/list/states", params=AQS_PARAMS)
        try:
            httpx.Response(500, request=request).raise_for_status()
        except httpx.HTTPStatusError:
            uvicorn_logger.exception("Exception in ASGI application")

        self.assert_credentials_redacted(uvicorn_stream.getvalue())


if __name__ == "__main__":
    unittest.main()
