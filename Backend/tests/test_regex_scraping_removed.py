import os
import sys
import unittest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)  # TODO: remove after pyproject.toml install


class RegexPathScrapingRemovedTests(unittest.TestCase):
    def test_message_utils_no_longer_defines_a_png_path_regex(self):
        import utils.message_utils as message_utils

        self.assertFalse(hasattr(message_utils, "PNG_PATH_RE"))

    def test_streaming_module_no_longer_scrapes_png_paths_from_tool_results(self):
        import inspect

        import utils.streaming as streaming

        source = inspect.getsource(streaming)
        self.assertNotIn(".png", source)

    def test_chat_stream_service_no_longer_references_png_path_regex(self):
        import inspect

        import services.chat_stream_service as chat_stream_service

        source = inspect.getsource(chat_stream_service)
        self.assertNotIn("PNG_PATH_RE", source)


if __name__ == "__main__":
    unittest.main()
