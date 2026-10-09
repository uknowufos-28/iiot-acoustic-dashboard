"""API contract tests that run without loading the TensorFlow Lite model."""

from __future__ import annotations

from dataclasses import asdict
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np

from acoustic_model.inference import WindowResult
from raspberry_pi.hybrid_api import create_server


TOKEN = "test-token-with-at-least-32-characters"
ORIGIN = "https://uknowufos-28.github.io"


class HybridApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(
            "127.0.0.1",
            0,
            model=object(),
            api_token=TOKEN,
            allowed_origins={ORIGIN},
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path: str, method: str = "GET", payload: dict | None = None, token: str = TOKEN, origin: str = ORIGIN):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Origin": origin}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if data is not None:
            headers["Content-Type"] = "application/json"
        return Request(self.base_url + path, data=data, headers=headers, method=method)

    def test_health_requires_token_and_returns_service_status(self) -> None:
        with urlopen(self.request("/api/health")) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
            self.assertEqual(json.load(response), {"status": "ok", "model": "hybrid_multiclass"})

        with self.assertRaises(HTTPError) as error:
            urlopen(self.request("/api/health", token="wrong"))
        self.assertEqual(error.exception.code, 401)

    def test_predict_validates_and_publishes_latest_result(self) -> None:
        result = WindowResult(
            window_number=1,
            start_seconds=0.0,
            end_seconds=3.0,
            label="Overhang fault",
            decision_scores={"Normal": -1.0, "Overhang fault": 0.5, "Underhang fault": 0.1},
        )
        payload = {"samples": [0.1, 0.2, 0.3], "sample_rate_hz": 16_000}
        with patch(
            "raspberry_pi.hybrid_api.analyze_signal",
            return_value=([result], np.asarray([0.1, 0.2, 0.3], dtype=np.float32)),
        ):
            with urlopen(self.request("/api/predict", "POST", payload)) as response:
                output = json.load(response)
                self.assertEqual(response.status, 200)
                self.assertEqual(output["windows"], [asdict(result)])

        with urlopen(self.request("/api/latest")) as response:
            latest = json.load(response)
        self.assertEqual(latest["received_at"], output["received_at"])
        self.assertEqual(latest["windows"][0]["label"], "Overhang fault")

    def test_prediction_rejects_non_finite_samples(self) -> None:
        request = self.request(
            "/api/predict",
            "POST",
            {"samples": [0.1, float("inf")], "sample_rate_hz": 16_000},
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_prediction_rejects_nested_samples_and_extreme_durations(self) -> None:
        for payload in (
            {"samples": [[0.1, 0.2]], "sample_rate_hz": 16_000},
            {"samples": [0.1], "sample_rate_hz": 999},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self.request("/api/predict", "POST", payload))
                self.assertEqual(error.exception.code, 400)

        too_long = {"samples": [0.0] * 60_001, "sample_rate_hz": 1_000}
        with self.assertRaises(HTTPError) as error:
            urlopen(self.request("/api/predict", "POST", too_long))
        self.assertEqual(error.exception.code, 413)

    def test_preflight_rejects_unconfigured_origins(self) -> None:
        request = self.request("/api/predict", "OPTIONS", origin="https://not-allowed.example")
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_server_requires_a_long_api_token(self) -> None:
        with self.assertRaises(ValueError):
            create_server("127.0.0.1", 0, object(), "short", {ORIGIN})


if __name__ == "__main__":
    unittest.main()
