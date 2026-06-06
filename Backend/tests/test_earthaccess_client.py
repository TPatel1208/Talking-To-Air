import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@unittest.skipIf(importlib.util.find_spec("earthaccess") is None, "earthaccess is not installed")
class EarthAccessClientTests(unittest.TestCase):
    def setUp(self):
        from utils import earthaccess_client

        earthaccess_client.reset_earthaccess_auth()

    def test_import_data_loader_does_not_authenticate(self):
        with patch("utils.earthaccess_client.earthaccess.login") as login:
            import preprocessing.data_loader as data_loader

            importlib.reload(data_loader)

        login.assert_not_called()

    def test_first_request_initializes_and_reuses_auth(self):
        from utils import earthaccess_client

        with patch(
            "utils.earthaccess_client.earthaccess.login",
            return_value=object(),
        ) as login:
            first = earthaccess_client.get_earthaccess_auth()
            second = earthaccess_client.get_earthaccess_auth()

        self.assertIs(first, second)
        self.assertEqual(login.call_count, 1)

    def test_login_falls_back_when_force_argument_is_unsupported(self):
        from utils import earthaccess_client

        auth = object()

        def fake_login(**kwargs):
            if "force" in kwargs:
                raise TypeError("login() got an unexpected keyword argument 'force'")
            return auth

        with patch("utils.earthaccess_client.earthaccess.login", side_effect=fake_login) as login:
            result = earthaccess_client.get_earthaccess_auth(force=True)

        self.assertIs(result, auth)
        self.assertEqual(login.call_count, 2)
        self.assertEqual(login.call_args.kwargs, {"strategy": "environment"})


if __name__ == "__main__":
    unittest.main()
