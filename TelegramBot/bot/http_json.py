"""Bounded HTTP JSON requests; redirects never move conversations elsewhere."""
import ctypes
import http.client
import json
import math
import os
import socket
import threading
import time
import urllib.parse
import urllib.request


def available_ram():
    if os.name != "nt":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Could not check available RAM")
    return status.avail_phys


def emergency_stop_model():
    req = urllib.request.Request("http://127.0.0.1:9090/api/stop",
                                 data=b'{"service":"llama-server"}',
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            response.read(4096)
    except Exception:
        pass


def request_json(url, payload=None, headers=None, timeout=10, guard_model=False,
                 cancel_event=None, include_status=False):
    """Read a bounded JSON response within one deadline, without redirects.

    Connection setup runs in a daemon because DNS resolution has no portable
    cancellation API. An expired connection can never send its payload.
    ``include_status`` lets API clients inspect structured non-2xx errors.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("HTTP timeout must be a positive finite number")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("Invalid HTTP endpoint")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("HTTP request cancelled")
    if guard_model:
        ram = available_ram()
        if ram is not None and ram < 1024**3:
            raise RuntimeError("Model request paused: less than 1 GiB available RAM")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    done = threading.Event()
    deadline = time.monotonic() + timeout
    failure = []
    sockets = []

    def monitor():
        low_since = None
        while not done.wait(.1):
            now = time.monotonic()
            reason = "HTTP request exceeded its total time limit" if now >= deadline else None
            if cancel_event is not None and cancel_event.is_set():
                reason = "HTTP request cancelled"
            if guard_model:
                try:
                    ram = available_ram()
                except Exception:
                    ram = None
                    reason = "RAM watchdog could not read available memory"
                if ram is not None and ram < .8 * 1024**3:
                    low_since = low_since or now
                    if now - low_since >= 3:
                        reason = "Low RAM watchdog stopped the model through the dashboard"
                else:
                    low_since = None
            if reason:
                failure.append(reason)
                for sock in sockets:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                if low_since and now - low_since >= 3:
                    emergency_stop_model()
                return

    watcher = threading.Thread(target=monitor, daemon=True, name="http-deadline")
    watcher.start()
    connected = threading.Event()
    connect_error = []

    def connect():
        try:
            conn.connect()
            sockets.append(conn.sock)
            if done.is_set() or failure:
                conn.close()
        except Exception as error:
            connect_error.append(error)
        finally:
            connected.set()

    try:
        threading.Thread(target=connect, daemon=True, name="http-connect").start()
        while not connected.wait(.05):
            if failure:
                raise RuntimeError(failure[0])
            if time.monotonic() >= deadline:
                raise RuntimeError("HTTP request exceeded its total time limit")
        if failure:
            raise RuntimeError(failure[0])
        if connect_error:
            raise connect_error[0]
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("HTTP request cancelled")
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        conn.request("POST" if body is not None else "GET", urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, "")), body, request_headers)
        response = conn.getresponse()
        if not include_status and not 200 <= response.status < 300:
            raise RuntimeError(f"HTTP {response.status}; endpoint rejected the request")
        data = response.read(4 * 1024 * 1024 + 1)
        if failure:
            raise RuntimeError(failure[0])
        if len(data) > 4 * 1024 * 1024:
            raise RuntimeError("HTTP response exceeded 4 MiB")
        decoded = json.loads(data.decode("utf-8"))
        return (response.status, decoded) if include_status else decoded
    except Exception as e:
        if failure:
            raise RuntimeError(failure[0]) from None
        if isinstance(e, RuntimeError):
            raise
        # Do not echo endpoint URLs, API keys or reflected response bodies.
        raise RuntimeError(f"HTTP request failed ({type(e).__name__})") from None
    finally:
        done.set()
        conn.close()
        watcher.join(timeout=.2)
