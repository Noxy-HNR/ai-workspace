"""Lifecycle endpoint coverage using an ephemeral loopback server only.

Shutdown is replaced by a recorder, so these tests never stop a live bot,
contact Telegram, or load/restart a model.
"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import app


class ResponseOrderHandler(app.Handler):
    def _send(self, obj, status=200):
        super()._send(obj, status)
        self.server.response_log.append(("response", status, obj))


class ManagementEndpointTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ResponseOrderHandler)
        self.server.daemon_threads = True
        self.server.response_log = []
        self.real_shutdown = self.server.shutdown
        self.shutdown_called = threading.Event()

        def record_shutdown():
            self.server.response_log.append(("shutdown",))
            self.shutdown_called.set()

        self.shutdown_patch = mock.patch.object(
            self.server, "shutdown", side_effect=record_shutdown)
        self.shutdown_mock = self.shutdown_patch.start()
        self.addCleanup(self.cleanup_server)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.db = mock.Mock(name="database")
        self.poller = mock.Mock(name="poller")
        self.db_patch = mock.patch.object(app, "DB", self.db)
        self.poller_patch = mock.patch.object(app, "POLLER", self.poller)
        self.client_patch = mock.patch.object(app, "make_client")
        self.models_patch = mock.patch.object(app.llm_mod, "list_models")
        self.generate_patch = mock.patch.object(app.llm_mod, "chat_completion")
        self.db_patch.start()
        self.poller_patch.start()
        self.client = self.client_patch.start()
        self.models = self.models_patch.start()
        self.generate = self.generate_patch.start()
        for patcher in (self.db_patch, self.poller_patch,
                        self.client_patch, self.models_patch, self.generate_patch):
            self.addCleanup(patcher.stop)

    def cleanup_server(self):
        self.shutdown_patch.stop()
        self.real_shutdown()
        self.server.server_close()

    def request(self, path="/api/shutdown", body=None, headers=None, method="POST"):
        request_headers = {"X-Relay-CSRF": app.CSRF_TOKEN,
                           "Content-Type": "application/json"}
        request_headers.update(headers or {})
        data = json.dumps({} if body is None else body).encode() if method == "POST" else None
        request = urllib.request.Request(
            self.base + path, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def assert_no_external_work(self):
        self.client.assert_not_called()
        self.models.assert_not_called()
        self.generate.assert_not_called()
        self.poller.assert_not_called()
        self.assertEqual(self.poller.mock_calls, [])
        self.assertEqual(self.db.mock_calls, [])

    def test_shutdown_acknowledges_before_scheduling_server_stop(self):
        status, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True, "shutting_down": True})
        self.assertTrue(self.shutdown_called.wait(timeout=2),
                        "Shutdown was not scheduled after the response")
        self.assertEqual(self.server.response_log,
                         [("response", 200, body), ("shutdown",)])
        self.shutdown_mock.assert_called_once_with()
        self.assert_no_external_work()

    def test_shutdown_requires_current_csrf_token(self):
        for token in ("", "expired-token"):
            with self.subTest(token=token):
                status, body = self.request(headers={"X-Relay-CSRF": token})
                self.assertEqual(status, 403)
                self.assertIn("error", body)
        self.shutdown_mock.assert_not_called()
        self.assert_no_external_work()

    def test_shutdown_rejects_foreign_host_or_origin(self):
        for headers in (
            {"Host": "relay.example"},
            {"Origin": "https://relay.example"},
            {"Origin": "http://127.0.0.1:1"},
            {"Origin": "null"},
        ):
            with self.subTest(headers=headers):
                status, body = self.request(headers=headers)
                self.assertEqual(status, 403)
                self.assertIn("loopback", body["error"])
        self.shutdown_mock.assert_not_called()
        self.assert_no_external_work()

    def test_shutdown_accepts_localhost_same_origin_management(self):
        local = f"localhost:{self.server.server_port}"
        status, body = self.request(headers={"Host": local, "Origin": "http://" + local})
        self.assertEqual((status, body), (200, {"ok": True, "shutting_down": True}))
        self.assertTrue(self.shutdown_called.wait(timeout=2))
        self.shutdown_mock.assert_called_once_with()
        self.assert_no_external_work()

    def test_get_and_malformed_body_do_not_stop_server(self):
        self.assertEqual(self.request(method="GET")[0], 404)
        self.assertEqual(self.request(body=[])[0], 400)
        self.shutdown_mock.assert_not_called()
        self.assert_no_external_work()


if __name__ == "__main__":
    unittest.main()
