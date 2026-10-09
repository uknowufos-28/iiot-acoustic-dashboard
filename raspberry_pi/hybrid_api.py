"""Authenticated HTTP API for the integrated CNN+SVM model on Raspberry Pi."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from hybrid_inference import analyze_signal, load_hybrid_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "hybrid_multiclass" / "hybrid_model.tflite"
MAX_REQUEST_BYTES = 20 * 1024 * 1024
MAX_SAMPLES = 1_000_000
MAX_RECORDING_SECONDS = 60
MIN_SAMPLE_RATE = 1_000
MAX_SAMPLE_RATE = 384_000
LOGGER = logging.getLogger("hybrid-api")


class HybridApiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        model,
        api_token: str,
        allowed_origins: set[str],
    ):
        super().__init__(address, HybridRequestHandler)
        self.model = model
        self.api_token = api_token
        self.allowed_origins = allowed_origins
        self.latest_result: dict | None = None
        self.latest_lock = threading.Lock()


class HybridRequestHandler(BaseHTTPRequestHandler):
    server: HybridApiServer

    def log_message(self, format: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _allowed_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            self._write_json(403, {"error": "Origin is not allowed."})
            return False
        return True

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.api_token}"
        supplied = self.headers.get("Authorization", "")
        if not supplied.isascii() or not hmac.compare_digest(supplied, expected):
            self._write_json(401, {"error": "A valid bearer token is required."})
            return False
        return True

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        if origin not in self.server.allowed_origins:
            self._write_json(403, {"error": "Origin is not allowed."})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._allowed_origin() or not self._authorized():
            return
        if self.path == "/api/health":
            self._write_json(200, {"status": "ok", "model": "hybrid_multiclass"})
            return
        if self.path == "/api/latest":
            with self.server.latest_lock:
                latest = self.server.latest_result
            self._write_json(200, latest or {"available": False})
            return
        self._write_json(404, {"error": "Endpoint not found."})

    def do_POST(self) -> None:
        if not self._allowed_origin() or not self._authorized():
            return
        if self.path != "/api/predict":
            self._write_json(404, {"error": "Endpoint not found."})
            return
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self._write_json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._write_json(411, {"error": "A valid Content-Length header is required."})
            return
        if content_length <= 0:
            self._write_json(400, {"error": "Request body is empty."})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._write_json(413, {"error": "Request body exceeds the configured size limit."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._write_json(400, {"error": f"Invalid JSON request: {exc}"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "JSON request must be an object."})
            return

        samples = payload.get("samples")
        sample_rate = payload.get("sample_rate_hz")
        if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
            self._write_json(400, {
                "error": f"samples must be a non-empty list of at most {MAX_SAMPLES} values."
            })
            return
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE
        ):
            self._write_json(400, {
                "error": (
                    f"sample_rate_hz must be an integer from {MIN_SAMPLE_RATE} "
                    f"to {MAX_SAMPLE_RATE}."
                )
            })
            return
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in samples
        ):
            self._write_json(400, {"error": "samples must be a flat list of numeric values."})
            return
        try:
            signal = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            self._write_json(400, {"error": f"samples must be numeric: {exc}"})
            return
        if not np.isfinite(signal).all():
            self._write_json(400, {"error": "samples must all be finite numeric values."})
            return
        if signal.size / sample_rate > MAX_RECORDING_SECONDS:
            self._write_json(413, {
                "error": f"Recordings are limited to {MAX_RECORDING_SECONDS} seconds per request."
            })
            return

        try:
            results, resampled = analyze_signal(signal, self.server.model, sample_rate)
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return
        except Exception:
            LOGGER.exception("Hybrid inference failed.")
            self._write_json(500, {"error": "Hybrid inference failed; check the Pi service log."})
            return

        latest = {
            "available": True,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "sample_rate_hz": sample_rate,
            "processing_sample_rate_hz": 16_000,
            "sample_count": int(signal.size),
            "resampled_sample_count": int(resampled.size),
            "window_count": len(results),
            "windows": [
                {
                    "window_number": item.window_number,
                    "start_seconds": item.start_seconds,
                    "end_seconds": item.end_seconds,
                    "label": item.label,
                    "decision_scores": item.decision_scores,
                }
                for item in results
            ],
        }
        with self.server.latest_lock:
            self.server.latest_result = latest
        self._write_json(200, latest)


def create_server(
    host: str,
    port: int,
    model,
    api_token: str,
    allowed_origins: set[str],
) -> HybridApiServer:
    if len(api_token) < 32 or not api_token.isascii():
        raise ValueError("The API bearer token must be at least 32 ASCII characters.")
    if not allowed_origins or any(not origin.startswith(("http://", "https://")) for origin in allowed_origins):
        raise ValueError("Configure at least one valid HTTP or HTTPS CORS origin.")
    return HybridApiServer((host, port), model, api_token, allowed_origins)


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default=os.environ.get("IIOT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("IIOT_PORT", "8765")))
    parser.add_argument("--tls-cert", type=Path, default=os.environ.get("IIOT_TLS_CERT_FILE"))
    parser.add_argument("--tls-key", type=Path, default=os.environ.get("IIOT_TLS_KEY_FILE"))
    args = parser.parse_args()

    api_token = os.environ.get("IIOT_API_TOKEN", "")
    if len(api_token) < 32 or not api_token.isascii():
        parser.error("Set IIOT_API_TOKEN to a random secret with at least 32 ASCII characters.")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("Provide both --tls-cert and --tls-key, or neither.")
    if not _is_loopback(args.host) and not args.tls_cert:
        parser.error("Non-loopback binding requires TLS; use a local HTTPS reverse proxy otherwise.")

    origins = {
        origin.strip()
        for origin in os.environ.get(
            "IIOT_ALLOWED_ORIGINS",
            "https://uknowufos-28.github.io,http://localhost:8000,http://127.0.0.1:8000",
        ).split(",")
        if origin.strip()
    }
    try:
        model = load_hybrid_model(str(args.model.resolve()))
        server = create_server(args.host, args.port, model, api_token, origins)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"Could not start the hybrid API: {exc}")

    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)

    LOGGER.info("Serving authenticated hybrid API on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down hybrid API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
