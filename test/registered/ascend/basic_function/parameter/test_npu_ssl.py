"""
Test SSL/TLS server startup parameters on NPU.

Tests the following --ssl-* and --enable-http2 parameters:
  --ssl-keyfile, --ssl-certfile, --ssl-keyfile-password,
  --enable-ssl-refresh, --enable-http2

Dependencies:
  pip install watchfiles granian httpx[http2]

Coverage:
  TC1: Basic HTTPS server launch
  TC2: Encrypted private key with --ssl-keyfile-password
  TC3: Encrypted key with wrong password → crash
  TC4: SSL certificate hot-reload (--enable-ssl-refresh)
  TC5: ssl-refresh + multi-worker → warning + fallback
  TC6: HTTP/2 + SSL server launch [needs granian, httpx[http2]]

"""

import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from queue import Queue

import httpx
import requests
import requests.exceptions as req_exc
import urllib3

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)

# Suppress InsecureRequestWarning for self-signed certs in HTTPS health checks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_self_signed_cert():
    """Generate a self-signed cert/key pair via openssl.

    Returns (key_path, cert_path).  Caller is responsible for cleanup.
    """
    key_fd, key_path = tempfile.mkstemp(suffix=".pem", prefix="ssl_key_")
    cert_fd, cert_path = tempfile.mkstemp(suffix=".pem", prefix="ssl_cert_")
    os.close(key_fd)
    os.close(cert_fd)

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key_path,
            "-out",
            cert_path,
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return key_path, cert_path


def _generate_encrypted_key_cert(password="testssl123"):
    """Generate a password-protected self-signed cert/key pair via openssl.

    Returns (key_path, cert_path).  Caller is responsible for cleanup.
    """
    key_fd, key_path = tempfile.mkstemp(suffix=".pem", prefix="ssl_enc_key_")
    cert_fd, cert_path = tempfile.mkstemp(suffix=".pem", prefix="ssl_enc_cert_")
    os.close(key_fd)
    os.close(cert_fd)

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key_path,
            "-out",
            cert_path,
            "-days",
            "1",
            "-passout",
            f"pass:{password}",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return key_path, cert_path, password


def _find_free_port():
    """Return an available TCP port number (≤ 55535 to keep grpc_port valid)."""
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
            if port <= 55535:
                return port
    raise RuntimeError("Could not find a free port ≤ 55535 after 100 attempts")


def _launch_https_server(
    model,
    ssl_keyfile,
    ssl_certfile,
    extra_args=None,
    timeout=300,
    continue_capture=False,
    ssl_keyfile_password=None,
):
    """Launch an SGLang server with HTTPS and wait for it to be ready.

    Uses a background thread to read stdout (avoids blocking the main
    health-polling loop) and polls /health via HTTPS until the server
    responds.

    Returns (process, port, full_stdout).
    """
    extra_args = extra_args or []
    port = _find_free_port()

    command = [
        "sglang",
        "serve",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--ssl-keyfile",
        ssl_keyfile,
        "--ssl-certfile",
        ssl_certfile,
    ]
    if ssl_keyfile_password:
        command.extend(["--ssl-keyfile-password", ssl_keyfile_password])
    command.extend(extra_args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Background thread reads stdout line-by-line so the main thread
    # is free to poll the HTTPS health endpoint without blocking.
    stdout_queue = Queue()

    def _read_stdout():
        try:
            for line in iter(process.stdout.readline, ""):
                stdout_queue.put(line)
                sys.stdout.write(f"[sglang] {line}")
                sys.stdout.flush()
        finally:
            process.stdout.close()

    reader_thread = threading.Thread(target=_read_stdout, daemon=True)
    reader_thread.start()

    try:
        start = time.time()
        ready = False

        while time.time() - start < timeout:
            # Check if process exited prematurely
            if process.poll() is not None:
                # Drain any remaining queued lines
                lines = _drain_queue(stdout_queue)
                raise RuntimeError(
                    f"Server exited with code {process.returncode}.\n"
                    f"Output:\n{''.join(lines[-50:])}"
                )

            # Poll HTTPS health endpoint (non-blocking, short timeout)
            try:
                resp = requests.get(
                    f"https://127.0.0.1:{port}/health",
                    verify=False,
                    timeout=3,
                )
                if resp.status_code == 200:
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.5)

        if not ready:
            lines = _drain_queue(stdout_queue)
            raise RuntimeError(
                f"Server did not become ready within {timeout}s.\n"
                f"Last output:\n{''.join(lines[-30:])}"
            )

        # Drain remaining stdout lines after server is ready
        stdout_text = "".join(_drain_queue(stdout_queue))
        if continue_capture:
            return process, port, stdout_text, stdout_queue
        return process, port, stdout_text

    except Exception:
        kill_process_tree(process.pid)
        raise


def _drain_queue(q):
    """Drain all items from a Queue without blocking, return as list."""
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except Exception:
            break
    return items


def _wait_for_queue_text(q, expected_text, timeout=10):
    """Poll queue with timeout until expected_text appears, return collected lines."""
    lines = []
    start = time.time()
    while time.time() - start < timeout:
        _drain_queue_into(q, lines)
        if any(expected_text in line for line in lines):
            return lines
        time.sleep(0.2)
    return lines


def _drain_queue_into(q, items):
    """Drain queue into existing list."""
    while True:
        try:
            items.append(q.get_nowait())
        except Exception:
            break


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestSSLParams(CustomTestCase):
    """Test SSL/TLS and HTTP/2 server startup parameters on NPU.

    [Test Category] Parameter
    [Test Target] --ssl-keyfile, --ssl-certfile, --ssl-keyfile-password,
                  --enable-ssl-refresh, --enable-http2
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.key_path, cls.cert_path = _generate_self_signed_cert()

    @classmethod
    def tearDownClass(cls):
        for path in (cls.key_path, cls.cert_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # TC1: Basic HTTPS server launch  (P0)
    # ------------------------------------------------------------------

    def test_ssl_basic_https_launch(self):
        """TC1: Server starts with --ssl-keyfile + --ssl-certfile,
        log contains 'SSL enabled', HTTP plain is rejected,
        HTTPS /health and /generate work.
        """
        process = None
        try:
            process, port, stdout = _launch_https_server(
                self.model, self.key_path, self.cert_path
            )

            # Verify SSL enabled log
            self.assertIn(
                "SSL enabled", stdout, "Server log should contain 'SSL enabled'"
            )

            # Verify HTTP (plain) is rejected on the SSL port
            with self.assertRaises(
                requests.exceptions.ConnectionError,
                msg="HTTP should be rejected on SSL port",
            ):
                requests.get(f"http://127.0.0.1:{port}/health", timeout=5)

            # Verify HTTPS health check works (no cert verification)
            resp = requests.get(
                f"https://127.0.0.1:{port}/health", verify=False, timeout=10
            )
            self.assertEqual(resp.status_code, 200)

            # Verify HTTPS with cert verification → self-signed cert trusted via cert_path
            resp = requests.get(
                f"https://127.0.0.1:{port}/health",
                verify=self.cert_path,
                timeout=10,
            )
            self.assertEqual(resp.status_code, 200)

            # Verify default CA verification rejects self-signed cert
            with self.assertRaises(
                req_exc.SSLError,
                msg="Self-signed cert should be rejected by default CA",
            ):
                requests.get(
                    f"https://127.0.0.1:{port}/health",
                    verify=True,
                    timeout=5,
                )

            # Verify end-to-end HTTPS inference
            resp = requests.post(
                f"https://127.0.0.1:{port}/generate",
                json={
                    "text": "Hello, who are you?",
                    "sampling_params": {"max_new_tokens": 50},
                },
                verify=self.cert_path,
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn(
                "text", body, "/generate response should contain 'text' field"
            )
            self.assertTrue(
                body["text"],
                f"Generated text should be non-empty, got {body.get('text')!r}",
            )

        finally:
            if process is not None:
                kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC2: Encrypted private key (ssl-keyfile-password)  (P0)
    # ------------------------------------------------------------------

    def test_ssl_encrypted_key(self):
        """TC2: --ssl-keyfile-password with encrypted private key,
        server starts and HTTPS /health and /generate work.

        Does NOT cover: granian + ssl_keyfile_password.
        """
        enc_key, enc_cert, password = _generate_encrypted_key_cert()
        process = None
        try:
            process, port, stdout = _launch_https_server(
                self.model,
                enc_key,
                enc_cert,
                ssl_keyfile_password=password,
            )

            self.assertIn(
                "SSL enabled", stdout, "Server log should contain 'SSL enabled'"
            )

            # HTTP plain must be rejected
            with self.assertRaises(
                requests.exceptions.ConnectionError,
                msg="HTTP should be rejected on SSL port",
            ):
                requests.get(f"http://127.0.0.1:{port}/health", timeout=5)

            # HTTPS health check (no cert verification)
            resp = requests.get(
                f"https://127.0.0.1:{port}/health", verify=False, timeout=10
            )
            self.assertEqual(resp.status_code, 200)

            # HTTPS with cert verification → self-signed cert trusted via cert_path
            resp = requests.get(
                f"https://127.0.0.1:{port}/health",
                verify=enc_cert,
                timeout=10,
            )
            self.assertEqual(resp.status_code, 200)

            # Default CA verification rejects self-signed cert
            with self.assertRaises(
                requests.exceptions.SSLError,
                msg="Self-signed cert should be rejected by default CA",
            ):
                requests.get(
                    f"https://127.0.0.1:{port}/health",
                    verify=True,
                    timeout=5,
                )

            # End-to-end HTTPS inference with encrypted key
            resp = requests.post(
                f"https://127.0.0.1:{port}/generate",
                json={"text": "Hello", "sampling_params": {"max_new_tokens": 16}},
                verify=enc_cert,
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn("text", body, "/generate response should contain 'text'")
            self.assertTrue(
                body["text"],
                f"Generated text should be non-empty, got {body.get('text')!r}",
            )

        finally:
            if process is not None:
                kill_process_tree(process.pid)
            for p in (enc_key, enc_cert):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # TC3: Encrypted key with wrong password → crash  (P1)
    # ------------------------------------------------------------------

    def test_ssl_encrypted_key_wrong_password(self):
        """TC3: --ssl-keyfile-password wrong for encrypted private key,
        server fails to start.
        """
        enc_key, enc_cert, _ = _generate_encrypted_key_cert()
        try:
            with self.assertRaises(
                RuntimeError,
                msg="Server with wrong password should fail to start",
            ):
                _launch_https_server(
                    self.model,
                    enc_key,
                    enc_cert,
                    ssl_keyfile_password="wrong_password",
                    timeout=30,
                )
        finally:
            for p in (enc_key, enc_cert):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # TC4: SSL certificate hot-reload (--enable-ssl-refresh)  (P0)
    # ------------------------------------------------------------------

    def test_ssl_enable_refresh(self):
        """TC4: --enable-ssl-refresh starts SSLCertRefresher,
        log contains 'auto-refresh enabled', and replacing cert
        files triggers hot-reload.

        Dependencies: watchfiles (pip install watchfiles)
        """
        process = None
        try:
            process, port, stdout, queue = _launch_https_server(
                self.model,
                self.key_path,
                self.cert_path,
                extra_args=["--enable-ssl-refresh"],
                continue_capture=True,
            )

            # Verify SSLCertRefresher started
            self.assertIn(
                "auto-refresh enabled",
                stdout,
                "Server log should contain SSL auto-refresh message",
            )

            # HTTPS still works under ssl-refresh
            resp = requests.get(
                f"https://127.0.0.1:{port}/health", verify=self.cert_path, timeout=10
            )
            self.assertEqual(resp.status_code, 200)

            # Verify end-to-end HTTPS inference
            resp = requests.post(
                f"https://127.0.0.1:{port}/generate",
                json={
                    "text": "Hello, who are you?",
                    "sampling_params": {"max_new_tokens": 50},
                },
                verify=self.cert_path,
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn(
                "text", body, "/generate response should contain 'text' field"
            )
            self.assertTrue(
                body["text"],
                f"Generated text should be non-empty, got {body.get('text')!r}",
            )

            # --- Hot-reload: replace cert files and verify reload ---
            new_key, new_cert = _generate_self_signed_cert()
            try:
                # Overwrite original cert/key with new files
                os.replace(new_key, self.key_path)
                os.replace(new_cert, self.cert_path)

                # Wait for watchfiles to detect change and reload
                lines = _wait_for_queue_text(
                    queue, "SSL cert/key reloaded successfully", timeout=15
                )
                self.assertTrue(
                    any("SSL cert/key reloaded successfully" in line for line in lines),
                    f"Cert reload not detected in logs within 15s",
                )

                # HTTPS still works after reload with new certs
                resp = requests.get(
                    f"https://127.0.0.1:{port}/health",
                    verify=self.cert_path,
                    timeout=10,
                )
                self.assertEqual(resp.status_code, 200)

                # Verify HTTPS inference still works after reload
                resp = requests.post(
                    f"https://127.0.0.1:{port}/generate",
                    json={
                        "text": "Hello after reload",
                        "sampling_params": {"max_new_tokens": 50},
                    },
                    verify=self.cert_path,
                    timeout=60,
                )
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertIn("text", body, "/generate response should contain 'text'")
                self.assertTrue(
                    body["text"],
                    f"Generated text should be non-empty, got {body.get('text')!r}",
                )

            finally:
                for p in (new_key, new_cert):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        finally:
            if process is not None:
                kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC5: ssl-refresh + multi-worker → warning + fallback  (P0)
    # ------------------------------------------------------------------

    def test_ssl_refresh_multi_worker_warning(self):
        """TC5: --enable-ssl-refresh --tokenizer-worker-num 2
        logs warning 'SSL refresh will be disabled' but static SSL works.
        """
        process = None
        try:
            process, port, stdout = _launch_https_server(
                self.model,
                self.key_path,
                self.cert_path,
                extra_args=[
                    "--enable-ssl-refresh",
                    "--tokenizer-worker-num",
                    "2",
                ],
            )

            # Verify warning was logged (multi-worker ssl-refresh fallback)
            self.assertIn(
                "SSL refresh will be disabled",
                stdout,
                "Server log should contain SSL refresh disabled warning",
            )

            # Static SSL should still work (fallback path)
            resp = requests.get(
                f"https://127.0.0.1:{port}/health", verify=self.cert_path, timeout=10
            )
            self.assertEqual(resp.status_code, 200)

            # Verify HTTPS inference still works in fallback mode
            resp = requests.post(
                f"https://127.0.0.1:{port}/generate",
                json={"text": "Hello", "sampling_params": {"max_new_tokens": 16}},
                verify=self.cert_path,
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn("text", body, "/generate response should contain 'text'")
            self.assertTrue(
                body["text"],
                f"Generated text should be non-empty, got {body.get('text')!r}",
            )

        finally:
            if process is not None:
                kill_process_tree(process.pid)

    # ------------------------------------------------------------------
    # TC6: HTTP/2 + SSL server launch  (P1)
    # ------------------------------------------------------------------

    def test_http2_ssl_launch(self):
        """TC6: --enable-http2 + SSL launches Granian with HTTPS,
        HTTP/2 protocol is negotiated, and HTTPS /health and /generate work.

        Does NOT cover: ssl_keyfile_password pass-through.

        Dependencies: granian, httpx[http2] (pip install granian httpx[http2])
        """
        process = None
        try:
            process, port, stdout = _launch_https_server(
                self.model,
                self.key_path,
                self.cert_path,
                extra_args=["--enable-http2"],
            )

            # Granian should log SSL enabled
            self.assertIn(
                "SSL enabled", stdout, "Granian log should contain 'SSL enabled'"
            )

            # HTTP (plain) must be rejected
            with self.assertRaises(
                requests.exceptions.ConnectionError,
                msg="HTTP should be rejected on SSL port",
            ):
                requests.get(f"http://127.0.0.1:{port}/health", timeout=5)

            # Use httpx with HTTP/2 support to verify protocol negotiation
            ssl_context = ssl.create_default_context(cafile=self.cert_path)
            with httpx.Client(http2=True, verify=ssl_context) as client:
                resp = client.get(f"https://127.0.0.1:{port}/health")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(
                    resp.http_version,
                    "HTTP/2",
                    f"Expected HTTP/2, got {resp.http_version}",
                )

                # Verify end-to-end HTTPS inference over HTTP/2
                resp = client.post(
                    f"https://127.0.0.1:{port}/generate",
                    json={"text": "Hello", "sampling_params": {"max_new_tokens": 16}},
                    timeout=60,
                )
                self.assertEqual(resp.status_code, 200)
                body = resp.json()
                self.assertIn("text", body, "/generate response should contain 'text'")
                self.assertTrue(
                    body["text"],
                    f"Generated text should be non-empty, got {body.get('text')!r}",
                )

        finally:
            if process is not None:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=3)
