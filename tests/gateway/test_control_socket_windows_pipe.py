"""Windows named-pipe transport for the gateway control socket (``platforms("windows")``).

The pipe client has no kernel read timeout: ``handle.read()`` blocks until the server writes.
#132547 — a multiplexer dialed its own pipe from its event loop, so the only process that could
answer was the one blocked in that read; the shutdown watchdog then killed the gateway (exit 75).
These pin the bounded client exchange: a server that never answers costs one ``None`` (callers
fall back to the scan layer), never an unbounded wedge, while a live server still answers.
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    _query_windows_pipe,
    query_gateway_control,
    windows_pipe_name,
)

pytestmark = pytest.mark.platforms("windows")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _request(verb: str = "identify") -> bytes:
    return json.dumps({"verb": verb, "id": 1, "protocol": CONTROL_PROTOCOL_VERSION}).encode("utf-8") + b"\n"


def _serve_pipe(home: Path, protocol_factory, ready: threading.Event) -> None:
    """Run a proactor pipe server in a background thread until the test ends."""

    async def main():
        loop = asyncio.get_running_loop()
        await loop.start_serving_pipe(protocol_factory, windows_pipe_name(home))
        ready.set()
        await asyncio.Event().wait()  # serve until the process exits

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert ready.wait(10), "pipe server did not start"


class _SilentProtocol(asyncio.Protocol):
    """Accepts the request and never answers — the wedge shape from #132547."""

    def connection_made(self, transport) -> None:  # pragma: no cover - exercised on windows
        self._transport = transport

    def data_received(self, data: bytes) -> None:  # pragma: no cover - exercised on windows
        pass  # deliberately no reply


def test_silent_pipe_server_costs_one_bounded_none(home):
    """A server that accepts and never replies must degrade to ``None`` within the timeout,
    not block the caller forever (the #132547 multiplexer kill)."""
    ready = threading.Event()
    _serve_pipe(home, _SilentProtocol, ready)

    started = time.monotonic()
    answer = _query_windows_pipe(home, _request(), timeout=1.0)
    elapsed = time.monotonic() - started

    assert answer is None
    # timeout (1.0) + the abandoned-worker join margin (0.5) + scheduler slack.
    assert elapsed < 4.0, f"bounded exchange took {elapsed:.2f}s"


def test_live_pipe_server_still_answers(home):
    """The bounded exchange must not break the working transport: the real server answers
    ``identify`` through the public client path."""
    server = GatewayControlServer(home)
    ready = threading.Event()

    async def main():
        loop = asyncio.get_running_loop()
        assert await server.start() is True
        ready.set()
        await asyncio.Event().wait()

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert ready.wait(10), "control server did not start"

    try:
        result = query_gateway_control(home, "identify", timeout=5.0)
    finally:
        server._pipe_server = None  # the daemon loop dies with the process

    assert isinstance(result, dict)
    assert result.get("pid") == os.getpid()
