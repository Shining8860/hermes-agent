"""Signal transport-retry contract: one retry owner, typed failures, no truncating success path.

Covers t_9f5fdb0c:
1. ``ConnectError`` on the first attempt -> one inline retry with the identical RPC payload,
   success, ``message_id`` stays ``None``.
2. ``ReadTimeout`` -> no automatic retry; the failure is typed ``retryable`` +
   ``delivery_uncertain`` so the delivery ledger owns the redelivery.
3. Daemon permanently gone -> bounded attempts and bounded wait, no loop.
4. ``send_voice``/``_send_attachment`` -> same semantics as text.
5. Rate-limit / JSON-RPC failure -> no transport retry, no plain-text fallback.
6. Notification path -> failure is logged as retryable instead of silently dropped.

No live signal-cli: every round-trip goes through ``httpx.MockTransport``.
"""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SUCCESS_BODY = {
    "jsonrpc": "2.0",
    "id": "1",
    "result": {"timestamp": 1234567890, "results": [{"type": "SUCCESS"}]},
}
RPC_ERROR_BODY = {
    "jsonrpc": "2.0",
    "id": "1",
    "error": {"code": -32603, "message": "internal failure"},
}


def _make_adapter(monkeypatch, account="+155****4567"):
    """A SignalAdapter whose typing indicator never consumes a transport call."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://signal.test", "account": account}
    adapter = SignalAdapter(config)
    adapter._stop_typing_indicator = AsyncMock()
    return adapter


def _recording_handler(outcomes):
    """MockTransport handler replaying *outcomes* (last one repeats) and recording RPC bodies.

    An outcome is an exception instance (raised from the transport) or ``(status, body)``.
    """
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        status, body = outcome
        return httpx.Response(status, json=body)

    return handler, calls


class TestTransportClassification:
    @pytest.mark.parametrize("exc,expected", [
        (httpx.ConnectError("connection refused"), False),   # provably undelivered
        (httpx.ConnectTimeout("connect timed out"), False),  # provably undelivered
        (httpx.ReadTimeout("read timed out"), True),         # ambiguous: may be delivered
        (httpx.WriteTimeout("write timed out"), True),       # ambiguous: may be delivered
        (httpx.RemoteProtocolError("peer closed"), True),    # ambiguous: may be delivered
        (ValueError("not a transport failure"), None),       # unchanged swallow-and-log path
    ])
    def test_only_undelivered_failures_are_retry_safe(self, exc, expected):
        from plugins.platforms.signal.adapter import _transport_failure_kind
        assert _transport_failure_kind(exc) is expected


class TestTextTransportRetry:
    @pytest.mark.asyncio
    async def test_connect_error_retried_once_with_identical_payload(self, monkeypatch):
        """Acceptance 1: an undelivered request is retried verbatim; no marker, no truncation."""
        adapter = _make_adapter(monkeypatch)
        handler, calls = _recording_handler([httpx.ConnectError("connection refused"), (200, SUCCESS_BODY)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("gateway.platforms.base.random.uniform", return_value=0.0):
                result = await adapter._send_with_retry(
                    "group:test-group", "hello world", base_delay=0.0)

        assert result.success is True
        assert result.message_id is None          # Signal has no editable message id
        assert result.error is None               # no warning prefix / fallback banner
        assert len(calls) == 2
        assert calls[0]["params"] == calls[1]["params"]
        assert calls[0]["params"]["message"] == "hello world"
        assert calls[1]["params"]["message"] == "hello world"

    @pytest.mark.asyncio
    async def test_read_timeout_is_not_retried_but_stays_retryable(self, monkeypatch):
        """Acceptance 2: ambiguous transport failure -> no inline retry, ledger-visible failure."""
        adapter = _make_adapter(monkeypatch)
        handler, calls = _recording_handler([httpx.ReadTimeout("read timed out")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("gateway.platforms.base.random.uniform", return_value=0.0):
                result = await adapter._send_with_retry(
                    "group:test-group", "hello world", base_delay=0.0)

        assert result.success is False
        assert result.retryable is True
        assert result.delivery_uncertain is True
        assert "ReadTimeout" in (result.error or "")
        assert len(calls) == 1                    # no duplicate risk taken

    @pytest.mark.asyncio
    async def test_permanently_down_daemon_is_bounded(self, monkeypatch):
        """Acceptance 3: 1 + max_retries attempts + one best-effort failure notice, bounded wait."""
        adapter = _make_adapter(monkeypatch)
        handler, calls = _recording_handler([httpx.ConnectError("connection refused")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("gateway.platforms.base.random.uniform", return_value=0.0):
                loop = asyncio.get_running_loop()
                started = loop.time()
                result = await adapter._send_with_retry(
                    "group:test-group", "hello world", max_retries=2, base_delay=0.05)
                elapsed = loop.time() - started

        assert result.success is False
        assert result.retryable is True
        # initial attempt + 2 retries + the final "delivery failed" notice
        assert len(calls) == 4
        # documented bound: base_delay * (2**max_retries - 1) + max_retries (+ slack)
        assert elapsed < 0.05 * (2 ** 2 - 1) + 2 + 0.5

    @pytest.mark.asyncio
    async def test_rpc_error_is_not_a_transport_retry(self, monkeypatch):
        """Acceptance 5a: a JSON-RPC-level refusal is final — no retry, no plain-text fallback."""
        adapter = _make_adapter(monkeypatch)
        handler, calls = _recording_handler([(200, RPC_ERROR_BODY)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("gateway.platforms.base.random.uniform", return_value=0.0):
                result = await adapter._send_with_retry(
                    "group:test-group", "hello world", base_delay=0.0)

        assert result.success is False
        assert result.retryable is False
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_response_is_not_a_transport_retry(self, monkeypatch):
        """Acceptance 5b: an HTTP 429 stays a plain refusal (SignalRateLimitError path untouched)."""
        adapter = _make_adapter(monkeypatch)
        handler, calls = _recording_handler([(429, {"error": "rate limited"})])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("gateway.platforms.base.random.uniform", return_value=0.0):
                result = await adapter._send_with_retry(
                    "group:test-group", "hello world", base_delay=0.0)

        assert result.success is False
        assert result.retryable is False
        assert len(calls) == 1


class TestAttachmentTransportRetry:
    @pytest.mark.asyncio
    async def test_voice_connect_error_retried_with_identical_payload(self, monkeypatch, tmp_path):
        """Acceptance 4: attachment sends get the same undelivered-retry semantics as text."""
        adapter = _make_adapter(monkeypatch)
        voice = tmp_path / "note.oga"
        voice.write_bytes(b"OggS" + b"\x00" * 64)
        handler, calls = _recording_handler([httpx.ConnectError("connection refused"), (200, SUCCESS_BODY)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            with patch("plugins.platforms.signal.adapter.SIGNAL_ATTACHMENT_RETRY_DELAY_SECONDS", 0):
                result = await adapter.send_voice("group:test-group", str(voice))

        assert result.success is True
        assert len(calls) == 2
        assert calls[0]["params"] == calls[1]["params"]
        assert calls[0]["params"]["attachments"] == [str(voice)]

    @pytest.mark.asyncio
    async def test_voice_read_timeout_is_reported_not_retried(self, monkeypatch, tmp_path):
        """Acceptance 4: an ambiguous attachment failure is surfaced typed, never re-sent inline."""
        adapter = _make_adapter(monkeypatch)
        voice = tmp_path / "note.oga"
        voice.write_bytes(b"OggS" + b"\x00" * 64)
        handler, calls = _recording_handler([httpx.ReadTimeout("read timed out")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter.client = client
            result = await adapter.send_voice("group:test-group", str(voice))

        assert result.success is False
        assert result.retryable is True
        assert result.delivery_uncertain is True
        assert "audio" in (result.error or "").lower()
        assert len(calls) == 1


class _StubAdapter(BasePlatformAdapter):
    """Minimal base adapter to prove the bracket's own delivery_uncertain short-circuit."""

    def __init__(self):
        super().__init__(PlatformConfig(), Platform.TELEGRAM)
        self.results = []
        self.calls = []

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.calls.append(content)
        return self.results.pop(0)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "direct", "chat_id": chat_id}


class TestBaseBracket:
    @pytest.mark.asyncio
    async def test_delivery_uncertain_never_reaches_the_plaintext_fallback(self):
        """Acceptance 4 (design): the 3,500-char fallback cannot turn an ambiguous send into success."""
        adapter = _StubAdapter()
        adapter.results = [SendResult(
            success=False, error="RPC send failed: ReadTimeout", retryable=True, delivery_uncertain=True)]
        result = await adapter._send_with_retry("chat1", "x" * 5000, max_retries=2, base_delay=0.0)

        assert result.success is False
        assert result.delivery_uncertain is True
        assert len(adapter.calls) == 1
        assert adapter.calls[0] == "x" * 5000     # untouched, never the 3,500-char prefix


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***", extra={})})
    adapter = MagicMock()
    adapter.name = "signal"
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m1"))
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True, message_id="p1"))
    adapter._send_with_retry = AsyncMock()
    runner.adapters = {Platform.SLACK: adapter}
    source = SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="channel", user_id="U123")
    return runner, adapter, SimpleNamespace(source=source)


class TestNotificationPath:
    @pytest.mark.asyncio
    async def test_notice_failure_is_logged_as_retryable(self, caplog):
        """Acceptance 6: a notice rides the retry bracket and its failure is visible."""
        runner, adapter, ctx = _make_runner()
        adapter._send_with_retry.return_value = SendResult(
            success=False, error="RPC send failed: ReadTimeout",
            retryable=True, delivery_uncertain=True)

        with caplog.at_level(logging.WARNING):
            await runner._deliver_platform_notice(ctx.source, "setup notice")

        adapter._send_with_retry.assert_awaited_once()
        adapter.send.assert_not_awaited()
        assert "Platform notice was not delivered" in caplog.text
        assert "retryable=True" in caplog.text
        assert "may-already-be-delivered=True" in caplog.text

    @pytest.mark.asyncio
    async def test_successful_notice_uses_the_retry_bracket_without_warning(self, caplog):
        runner, adapter, ctx = _make_runner()
        adapter._send_with_retry.return_value = SendResult(success=True, message_id="m2")

        with caplog.at_level(logging.WARNING):
            await runner._deliver_platform_notice(ctx.source, "setup notice")

        adapter._send_with_retry.assert_awaited_once()
        adapter.send.assert_not_awaited()
        assert "Platform notice was not delivered" not in caplog.text
