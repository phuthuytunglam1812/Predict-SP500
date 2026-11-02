import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import streamlit as st
from streamlit.testing.v1 import AppTest


class StartupTest(unittest.TestCase):
    def test_slow_market_providers_do_not_block_inputs_or_predictions(self):
        release_requests = threading.Event()
        request_started = {"cpi": threading.Event(), "spy": threading.Event(), "vix": threading.Event()}
        request_finished = {name: threading.Event() for name in request_started}

        def slow_provider(request, **kwargs):
            name = "cpi" if "bls.gov" in request.full_url else "vix" if "%5EVIX" in request.full_url else "spy"
            request_started[name].set()
            release_requests.wait(timeout=30)
            request_finished[name].set()
            raise URLError("Simulated slow market provider")

        # Test the fallback path without spending API credits or loading local secrets.
        offline_openai = types.SimpleNamespace(OpenAI=Mock(side_effect=RuntimeError("Offline test")))
        st.cache_resource.clear()
        st.cache_data.clear()
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        try:
            with patch("urllib.request.urlopen", side_effect=slow_provider) as fetch, patch.dict(sys.modules, {"openai": offline_openai}):
                app = AppTest.from_file(str(app_path), default_timeout=20).run()
                self.assertFalse(app.exception, app.exception)
                fetch.assert_not_called()
                self.assertNotIn("predict_helper", sys.modules)
                self.assertEqual(app.button[0].label, "Start Analysis")

                started_at = time.monotonic()
                app.button[0].click().run()
                elapsed = time.monotonic() - started_at
                self.assertFalse(app.exception, app.exception)
                self.assertLess(elapsed, 10, "Input controls waited for external providers")
                self.assertEqual(len(app.sidebar.number_input), 3)
                self.assertNotIn("sklearn", sys.modules)
                self.assertNotIn("xgboost", sys.modules)
                for name, event in request_started.items():
                    self.assertTrue(event.wait(timeout=2), f"{name} was not fetched concurrently")
                self.assertFalse(release_requests.is_set())
                self.assertTrue(any("BLS updating" in block.value for block in app.markdown))

                confirm = next(button for button in app.button if button.label == "Confirm & Run Models")
                confirm.click().run()
                self.assertFalse(app.exception, app.exception)
                analysis = app.session_state["analysis"]
                self.assertFalse(analysis["result"]["used_fallback"])
                self.assertEqual(analysis["allocation"]["stocks_pct"] + analysis["allocation"]["bonds_pct"], 100)
                self.assertAlmostEqual(analysis["stocks_amount"] + analysis["bonds_amount"], analysis["portfolio_amount"])
                print(f"Inputs available in {elapsed:.2f}s while all three market requests were blocked")
        finally:
            release_requests.set()
            for name, event in request_finished.items():
                if request_started[name].is_set():
                    event.wait(timeout=2)
            st.cache_resource.clear()
            st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
