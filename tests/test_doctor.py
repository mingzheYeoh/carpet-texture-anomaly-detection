"""Small diagnostic contract: failures must remain visible without dependencies."""

import importlib.util
import unittest
from unittest.mock import patch


class DoctorCheck(unittest.TestCase):
    def test_missing_import_is_reported_and_returns_failure(self):
        self.assertIsNotNone(importlib.util.find_spec("carpet_ad.doctor"))
        from carpet_ad.doctor import diagnose

        with patch("carpet_ad.doctor.importlib.import_module", side_effect=ImportError("missing dependency")):
            report = diagnose()
        self.assertFalse(report["ok"])
        self.assertIn("missing dependency", report["imports"]["numpy"]["error"])
        self.assertEqual(report["cuda"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
