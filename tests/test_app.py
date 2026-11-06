"""Smoke test that renders the Streamlit app with its testing harness."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_and_predicts(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

        self.assertEqual([error.value for error in app.exception], [])
        self.assertTrue(app.button, "expected the predict button to be rendered")

        app.button[0].click().run()

        self.assertEqual([error.value for error in app.exception], [])
        self.assertTrue(
            any("Predicted traffic" in header.value for header in app.subheader),
            "prediction result was not rendered",
        )
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual([tab.label for tab in app.tabs], ["Overview", "Agent outputs", "Communication log"])


if __name__ == "__main__":
    unittest.main()
