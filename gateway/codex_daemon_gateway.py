"""Bridge Discord threads directly to tasks on a shared Codex app-server daemon.

The bridge uses Hermes only for Discord transport. Codex owns every task,
transcript, tool call, permission decision, and model response.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Awaitable, Callable, Optional

import aiohttp
import yaml

from gateway.platforms.base import _prefix_within_utf16_limit, utf16_len
from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

_INITIALIZE_TIMEOUT_SECONDS = 15.0
_REQUEST_TIMEOUT_SECONDS = 45.0
_RECONNECT_MAX_DELAY_SECONDS = 30.0
_ROLLOUT_RETRY_INITIAL_DELAY_SECONDS = 0.5
_ROLLOUT_RETRY_MAX_DELAY_SECONDS = 30.0
_STREAM_EDIT_INTERVAL_SECONDS = 0.75
_HISTORY_PAGE_SIZE = 100
_DISCORD_THREAD_TITLE_MAX_UTF16_UNITS = 80
_DISCORD_THREAD_TITLE_SEPARATOR = " | "
_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class CodexRpcError(RuntimeError):
    """An error response returned by the Codex app-server."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _is_missing_rollout_error(error: CodexRpcError, thread_id: str) -> bool:
    """Match the transient resume error emitted before a rollout is durable."""
    expected = f"no rollout found for thread id {thread_id}"
    return error.message == expected


def _truncate_utf16_prefix(text: str, limit: int) -> str:
    """Truncate a title component from the right within a UTF-16 budget."""
    if utf16_len(text) <= limit:
        return text
    if limit <= 3:
        return _prefix_within_utf16_limit(text, limit)
    return _prefix_within_utf16_limit(text, limit - 3).rstrip() + "..."


def _truncate_utf16_suffix(text: str, limit: int) -> str:
    """Truncate a path label from the left within a UTF-16 budget."""
    if utf16_len(text) <= limit:
        return text
    if limit <= 3:
        return _prefix_within_utf16_limit(text[::-1], limit)[::-1]
    suffix = _prefix_within_utf16_limit(text[::-1], limit - 3)[::-1].lstrip()
    return "..." + suffix


def _parent_and_working_directory(cwd: str) -> str:
    """Return the final two path components for a Codex working directory."""
    path = Path(str(cwd or "."))
    parts = list(path.parts)
    if path.anchor and parts and parts[0] == path.anchor:
        parts = parts[1:]
    if not parts:
        return path.anchor or "."
    return "/".join(parts[-2:])


def _discord_thread_title(
    name: str,
    cwd: str,
    codex_thread_id: str,
    *,
    working: bool = False,
) -> str:
    """Render a status-prefixed task identity within Discord's title limit."""
    cleaned_name = re.sub(r"\s+", " ", str(name or "Untitled task")).strip()
    cleaned_name = cleaned_name or "Untitled task"
    directory = re.sub(
        r"\s+", " ", _parent_and_working_directory(cwd)
    ).strip()
    session_id = str(codex_thread_id or "").strip()
    if not session_id:
        raise ValueError("A Discord Codex thread title requires a session id")

    status_prefix = "⏳ " if working else "✅ "
    separator_units = utf16_len(_DISCORD_THREAD_TITLE_SEPARATOR)
    fixed_units = (
        utf16_len(status_prefix) + utf16_len(session_id) + 2 * separator_units
    )
    # Reserve two UTF-16 units so any first Unicode code point can remain.
    directory_budget = _DISCORD_THREAD_TITLE_MAX_UTF16_UNITS - fixed_units - 2
    if directory_budget < 1:
        raise ValueError("The Codex session id exceeds the Discord title budget")
    directory = _truncate_utf16_suffix(directory, directory_budget)
    suffix = (
        f"{_DISCORD_THREAD_TITLE_SEPARATOR}{directory}"
        f"{_DISCORD_THREAD_TITLE_SEPARATOR}{session_id}"
    )
    name_budget = (
        _DISCORD_THREAD_TITLE_MAX_UTF16_UNITS
        - utf16_len(status_prefix)
        - utf16_len(suffix)
    )
    displayed_name = _truncate_utf16_prefix(cleaned_name, name_budget)
    return f"{status_prefix}{displayed_name}{suffix}"


@dataclass(frozen=True)
class CodexGatewaySettings:
    """Configuration for one local Discord-to-Codex daemon bridge."""

    enabled: bool
    socket_path: Path
    default_cwd: Path
    state_path: Path
    parent_chat_id: Optional[str] = None
    member_user_ids: tuple[str, ...] = ()

    @classmethod
    def load(cls) -> "CodexGatewaySettings":
        """Load the bridge settings from the active Hermes home directory."""
        hermes_home = Path(get_hermes_home())
        config_path = hermes_home / "config.yaml"
        raw: dict[str, Any] = {}
        if config_path.exists():
            parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        section = raw.get("codex_gateway")
        if not isinstance(section, dict):
            section = {}
        raw_member_user_ids = section.get("member_user_ids", [])
        if not isinstance(raw_member_user_ids, list):
            raise ValueError("codex_gateway.member_user_ids must be a YAML list")
        socket_path = Path(
            str(
                section.get("socket_path")
                or Path.home()
                / ".codex"
                / "app-server-control"
                / "app-server-control.sock"
            )
        ).expanduser()
        default_cwd = Path(str(section.get("default_cwd") or Path.home())).expanduser()
        state_path = hermes_home / "state" / "codex_gateway_discord_threads.json"
        return cls(
            enabled=bool(section.get("enabled", False)),
            socket_path=socket_path,
            default_cwd=default_cwd,
            state_path=state_path,
            parent_chat_id=(
                str(section["parent_chat_id"])
                if section.get("parent_chat_id")
                else None
            ),
            member_user_ids=tuple(
                str(user_id)
                for user_id in raw_member_user_ids
                if str(user_id)
            ),
        )


@dataclass
class CodexItemDelivery:
    """Durable Discord delivery cursor for one Codex transcript item."""

    discord_message_id: Optional[str] = None
    final: bool = False
    text: str = ""
    prefix: str = ""
    suffix: str = ""


@dataclass
class CodexTaskBinding:
    """Durable relation between one Discord thread and one Codex task."""

    discord_chat_id: str
    codex_thread_id: str
    cwd: str
    title: str = ""
    discord_title: str = ""
    guild_id: Optional[str] = None
    parent_chat_id: Optional[str] = None
    item_deliveries: dict[str, CodexItemDelivery] = field(default_factory=dict)
    pending_discord_message_ids: list[str] = field(default_factory=list)


class CodexTaskBindingStore:
    """Persist Discord-thread mappings without storing Codex transcript data."""

    def __init__(self, path: Path):
        self.path = path
        self.bindings: dict[str, CodexTaskBinding] = {}
        self._load()

    def _load(self) -> None:
        """Load valid mappings and ignore malformed records independently."""
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read Codex gateway task mappings")
            return
        records = payload.get("bindings", {}) if isinstance(payload, dict) else {}
        if not isinstance(records, dict):
            return
        for chat_id, record in records.items():
            if not isinstance(record, dict):
                continue
            try:
                binding_data = dict(record)
                raw_deliveries = binding_data.pop("item_deliveries", {})
                deliveries = {
                    str(item_id): CodexItemDelivery(**delivery)
                    for item_id, delivery in raw_deliveries.items()
                    if isinstance(delivery, dict)
                }
                binding = CodexTaskBinding(
                    **binding_data,
                    item_deliveries=deliveries,
                )
            except TypeError:
                logger.warning("Ignoring malformed Codex gateway mapping for %s", chat_id)
                continue
            if binding.discord_chat_id == str(chat_id) and binding.codex_thread_id:
                self.bindings[str(chat_id)] = binding

    def bind(self, binding: CodexTaskBinding) -> None:
        """Replace the mapping for one Discord thread and persist it atomically."""
        for chat_id, stored in list(self.bindings.items()):
            if (
                chat_id != binding.discord_chat_id
                and stored.codex_thread_id == binding.codex_thread_id
            ):
                self.bindings.pop(chat_id)
        self.bindings[binding.discord_chat_id] = binding
        self._save()

    def unbind(self, discord_chat_id: str) -> Optional[CodexTaskBinding]:
        """Remove and return the mapping for one Discord thread."""
        binding = self.bindings.pop(discord_chat_id, None)
        if binding is not None:
            self._save()
        return binding

    def chats_for_codex_thread(self, codex_thread_id: str) -> list[str]:
        """Return Discord threads that are attached to a Codex task."""
        return [
            chat_id
            for chat_id, binding in self.bindings.items()
            if binding.codex_thread_id == codex_thread_id
        ]

    def item_delivery(
        self, discord_chat_id: str, item_id: str
    ) -> Optional[CodexItemDelivery]:
        """Return the stored Discord delivery cursor for one Codex item."""
        binding = self.bindings.get(discord_chat_id)
        return binding.item_deliveries.get(item_id) if binding else None

    def record_item_delivery(
        self,
        discord_chat_id: str,
        item_id: str,
        delivery: CodexItemDelivery,
    ) -> None:
        """Persist a newer Discord delivery cursor without regressing final state."""
        binding = self.bindings.get(discord_chat_id)
        if binding is None or not item_id:
            return
        stored = binding.item_deliveries.get(item_id)
        if stored == delivery or (stored is not None and stored.final and not delivery.final):
            return
        binding.item_deliveries[item_id] = delivery
        self._save()

    def set_discord_message_pending(
        self,
        discord_chat_id: str,
        message_id: str,
        *,
        pending: bool,
    ) -> None:
        """Persist whether a Discord prompt still awaits its Codex user item."""
        binding = self.bindings.get(discord_chat_id)
        if binding is None or not message_id:
            return
        message_ids = set(binding.pending_discord_message_ids)
        if pending:
            message_ids.add(message_id)
        else:
            message_ids.discard(message_id)
        updated = sorted(message_ids)
        if updated == binding.pending_discord_message_ids:
            return
        binding.pending_discord_message_ids = updated
        self._save()

    def _save(self) -> None:
        """Write mappings with a same-directory atomic replacement."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "bindings": {
                chat_id: asdict(binding)
                for chat_id, binding in sorted(self.bindings.items())
            },
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)


EventHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
ServerRequestHandler = Callable[[Any, str, dict[str, Any]], Awaitable[None]]
ConnectedHandler = Callable[[], Awaitable[None]]


class CodexDaemonClient:
    """Maintain an ordered JSON-RPC connection to a Codex daemon Unix socket."""

    def __init__(
        self,
        socket_path: Path,
        *,
        notification_handler: EventHandler,
        server_request_handler: ServerRequestHandler,
        connected_handler: ConnectedHandler,
    ):
        """Configure callbacks and initialize disconnected client state."""
        self.socket_path = socket_path
        self.notification_handler = notification_handler
        self.server_request_handler = server_request_handler
        self.connected_handler = connected_handler
        self._connected = asyncio.Event()
        self._closed = False
        self._request_counter = 0
        self._pending_requests: dict[Any, asyncio.Future[Any]] = {}
        self._send_lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[tuple[str, Any, str, dict[str, Any]]] = (
            asyncio.Queue()
        )
        self._supervisor_task: Optional[asyncio.Task[None]] = None
        self._dispatcher_task: Optional[asyncio.Task[None]] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None

    async def start(self) -> None:
        """Start connection supervision and wait for the first initialized socket."""
        if self._supervisor_task is not None:
            return
        self._dispatcher_task = asyncio.create_task(
            self._dispatch_events(), name="codex-daemon-event-dispatcher"
        )
        self._supervisor_task = asyncio.create_task(
            self._supervise(), name="codex-daemon-connection-supervisor"
        )
        try:
            await asyncio.wait_for(
                self._connected.wait(), timeout=_INITIALIZE_TIMEOUT_SECONDS
            )
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Stop reconnect attempts and fail every unresolved request."""
        if self._closed:
            return
        self._closed = True
        self._connected.clear()
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("Codex daemon connection closed"))
        self._pending_requests.clear()
        websocket = self._websocket
        if websocket is not None and not websocket.closed:
            await websocket.close()
        for task in (self._supervisor_task, self._dispatcher_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        for task in (self._supervisor_task, self._dispatcher_task):
            if task is not None and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError):
                    await task
        await self._close_transport()

    async def request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        *,
        timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> Any:
        """Send a JSON-RPC request and return its result object."""
        await asyncio.wait_for(self._connected.wait(), timeout=_INITIALIZE_TIMEOUT_SECONDS)
        self._request_counter += 1
        request_id = self._request_counter
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        try:
            await self._send(payload)
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except BaseException:
            self._pending_requests.pop(request_id, None)
            raise

    async def respond(self, request_id: Any, result: Any) -> None:
        """Resolve one app-server request with a successful result."""
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def reject(self, request_id: Any, code: int, message: str) -> None:
        """Resolve one app-server request with a JSON-RPC error."""
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        """Serialize one WebSocket write against connection replacement."""
        async with self._send_lock:
            websocket = self._websocket
            if websocket is None or websocket.closed:
                raise ConnectionError("Codex daemon is disconnected")
            await websocket.send_json(payload)

    async def _supervise(self) -> None:
        """Reconnect with bounded backoff and rejoin mapped Codex tasks."""
        delay_seconds = 1.0
        while not self._closed:
            try:
                await self._connect_and_read()
                delay_seconds = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Codex daemon connection failed")
            finally:
                self._connected.clear()
                self._fail_pending_requests()
                await self._close_transport()
            if not self._closed:
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(
                    _RECONNECT_MAX_DELAY_SECONDS, delay_seconds * 2.0
                )

    async def _connect_and_read(self) -> None:
        """Initialize one Unix-socket WebSocket and consume it until disconnect."""
        connector = aiohttp.UnixConnector(path=str(self.socket_path))
        session = aiohttp.ClientSession(connector=connector)
        self._session = session
        websocket = await session.ws_connect(
            "http://localhost/rpc",
            max_msg_size=128 << 20,
            compress=0,
            heartbeat=20.0,
        )
        initialize_id = "codex-discord-gateway-initialize"
        await websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": initialize_id,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "hermes_codex_discord_gateway",
                        "version": "0.1.0",
                    },
                    "capabilities": {
                        "experimentalApi": True,
                        "requestAttestation": False,
                    },
                },
            }
        )
        await self._await_initialize(websocket, initialize_id)
        self._websocket = websocket
        self._connected.set()
        reader_task = asyncio.create_task(
            self._read_messages(websocket), name="codex-daemon-websocket-reader"
        )
        try:
            await self.connected_handler()
            await reader_task
        finally:
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task

    async def _await_initialize(
        self, websocket: aiohttp.ClientWebSocketResponse, initialize_id: str
    ) -> None:
        """Require a successful initialize response before exposing the socket."""
        async def receive_initialize() -> None:
            """Read frames until the initialize request has a terminal response."""
            while True:
                message = await websocket.receive()
                if message.type != aiohttp.WSMsgType.TEXT:
                    if message.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        raise ConnectionError("Codex daemon closed during initialize")
                    continue
                payload = json.loads(message.data)
                if payload.get("id") != initialize_id:
                    continue
                if "error" in payload:
                    error = payload["error"]
                    raise CodexRpcError(
                        int(error.get("code", -32000)),
                        str(error.get("message", "initialize failed")),
                        error.get("data"),
                    )
                return

        await asyncio.wait_for(
            receive_initialize(), timeout=_INITIALIZE_TIMEOUT_SECONDS
        )

    async def _read_messages(
        self, websocket: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Route responses immediately and enqueue ordered server events."""
        async for message in websocket:
            if message.type == aiohttp.WSMsgType.TEXT:
                payload = json.loads(message.data)
                request_id = payload.get("id")
                method = payload.get("method")
                if method is not None and request_id is not None:
                    await self._event_queue.put(
                        ("request", request_id, str(method), payload.get("params") or {})
                    )
                elif method is not None:
                    await self._event_queue.put(
                        ("notification", None, str(method), payload.get("params") or {})
                    )
                elif request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)
                    if "error" in payload:
                        error = payload["error"]
                        future.set_exception(
                            CodexRpcError(
                                int(error.get("code", -32000)),
                                str(error.get("message", "Codex request failed")),
                                error.get("data"),
                            )
                        )
                    else:
                        future.set_result(payload.get("result"))
            elif message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                break
        if not self._closed:
            raise ConnectionError("Codex daemon WebSocket disconnected")

    async def _dispatch_events(self) -> None:
        """Apply notifications and requests in the order received on the socket."""
        while True:
            kind, request_id, method, params = await self._event_queue.get()
            try:
                if kind == "notification":
                    await self.notification_handler(method, params)
                else:
                    await self.server_request_handler(request_id, method, params)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Codex gateway event handler failed for %s", method)
            finally:
                self._event_queue.task_done()

    def _fail_pending_requests(self) -> None:
        """Wake request callers when the connection that owned them disappears."""
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(ConnectionError("Codex daemon disconnected"))
        self._pending_requests.clear()

    async def _close_transport(self) -> None:
        """Close the active WebSocket and its Unix-connector session."""
        websocket = self._websocket
        self._websocket = None
        if websocket is not None and not websocket.closed:
            await websocket.close()
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()


@dataclass
class PendingCodexServerRequest:
    """A Codex server request awaiting an answer from one Discord thread."""

    request_id: Any
    method: str
    params: dict[str, Any]


@dataclass
class DiscordStreamMessage:
    """Cumulative text and Discord delivery state for one Codex item."""

    turn_id: str = ""
    activity_kind: Optional[str] = None
    text: str = ""
    prefix: str = ""
    suffix: str = ""
    message_id: Optional[str] = None
    flush_task: Optional[asyncio.Task[None]] = None


@dataclass
class DiscordActivityDetail:
    """Expanded text and summary category for one turn activity item."""

    text: str
    kind: str
    failed: bool = False


@dataclass
class DiscordTurnDisplay:
    """One collapsed Discord activity message and its expandable details."""

    status_message_id: Optional[str] = None
    details: dict[str, DiscordActivityDetail] = field(default_factory=dict)
    status: str = "inProgress"
    legacy_split_messages: bool = False
    rendered_on_connection: bool = False


class DiscordCodexGateway:
    """Translate Discord input and Codex events without invoking Hermes AIAgent."""

    def __init__(
        self,
        adapter: Any,
        settings: CodexGatewaySettings,
        *,
        client: Optional[CodexDaemonClient] = None,
    ):
        """Load durable mappings and initialize task-scoped delivery state."""
        self.adapter = adapter
        self.settings = settings
        self.bindings = CodexTaskBindingStore(settings.state_path)
        self.active_turns: dict[str, str] = {}
        self.submitted_client_ids: set[str] = {
            message_id
            for binding in self.bindings.bindings.values()
            for message_id in binding.pending_discord_message_ids
        }
        self.pending_server_requests: dict[str, list[PendingCodexServerRequest]] = {}
        self.stream_messages: dict[tuple[str, str], DiscordStreamMessage] = {}
        self.turn_displays: dict[tuple[str, str], DiscordTurnDisplay] = {}
        self.item_phases: dict[tuple[str, str], str] = {}
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._pending_gateway_thread_starts = 0
        self._gateway_started_thread_ids: set[str] = set()
        self._deferred_started_threads: dict[str, dict[str, Any]] = {}
        self._provisional_thread_tasks: dict[str, asyncio.Task[None]] = {}
        self._awaiting_rollout_thread_ids: set[str] = set()
        self._history_syncing_thread_ids: set[str] = set()
        self._queued_history_notifications: dict[
            str, list[tuple[str, dict[str, Any]]]
        ] = {}
        self.client = client or CodexDaemonClient(
            settings.socket_path,
            notification_handler=self.handle_notification,
            server_request_handler=self.handle_server_request,
            connected_handler=self._resume_bound_tasks,
        )

    async def start(self) -> None:
        """Connect to the daemon and subscribe to every durable Discord mapping."""
        await self.client.start()
        logger.info(
            "Codex Discord gateway connected to %s with %d task mapping(s)",
            self.settings.socket_path,
            len(self.bindings.bindings),
        )

    async def stop(self) -> None:
        """Cancel delivery tasks and close the daemon connection."""
        await self._cancel_all_provisional_threads()
        tasks = [
            stream.flush_task
            for stream in self.stream_messages.values()
            if stream.flush_task is not None
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        await self.client.close()

    async def handle_message(self, event: Any) -> str:
        """Route one authorized Discord message to a Codex task or command."""
        chat_id = str(event.source.chat_id)
        command = event.get_command()
        if command and command.lower() == "refresh":
            return await self._refresh_task(chat_id)
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            if command:
                return await self._handle_command(event, command.lower())
            pending = self.pending_server_requests.get(chat_id, [])
            if pending and pending[0].method in {
                "item/tool/requestUserInput",
                "mcpServer/elicitation/request",
            }:
                return await self._answer_user_input(chat_id, event.text or "")
            binding = self.bindings.bindings.get(chat_id)
            if binding is None:
                binding = await self._create_task_for_event(event)
            await self._send_event_to_codex(binding, event)
            return ""

    async def _handle_command(self, event: Any, command: str) -> str:
        """Implement Codex task commands inside the Discord thread."""
        chat_id = str(event.source.chat_id)
        args = (event.get_command_args() or "").strip()
        if command in {"help", "codex"}:
            return self._help_text()
        if command in {"sessions", "tasks"}:
            return await self._list_tasks(args)
        if command == "resume":
            return await self._resume_from_command(event, args)
        if command in {"new", "reset"}:
            return await self._new_from_command(event, args)
        if command == "status":
            return await self._status_text(chat_id)
        if command in {"stop", "interrupt"}:
            return await self._interrupt_task(chat_id)
        if command in {"approve", "always", "deny", "cancel"}:
            approval_command = command
            if command == "approve" and args.lower() in {"session", "always"}:
                approval_command = "always"
            return await self._resolve_pending_request(chat_id, approval_command)
        if command == "answer":
            return await self._answer_user_input(chat_id, args)
        if command == "unlink":
            removed = self.bindings.unbind(chat_id)
            return "Codex task detached." if removed else "This Discord thread is not attached."
        return (
            f"Unknown Codex gateway command `/{command}`. "
            "Use `/help` for the available commands."
        )

    async def _create_task_for_event(self, event: Any) -> CodexTaskBinding:
        """Create, name, and bind a Codex task before its first Discord turn."""
        source = event.source
        result = await self._start_gateway_thread(
            {
                "cwd": str(self.settings.default_cwd),
                "historyMode": "paginated",
            },
        )
        thread = result.get("thread", {}) if isinstance(result, dict) else {}
        thread_id = str(thread.get("id") or "")
        if not thread_id:
            raise RuntimeError("Codex thread/start returned no task id")
        title = str(
            getattr(source, "auto_thread_initial_name", None)
            or getattr(source, "chat_name", None)
            or "Discord Codex task"
        )
        binding = CodexTaskBinding(
            discord_chat_id=str(source.chat_id),
            codex_thread_id=thread_id,
            cwd=str(result.get("cwd") or self.settings.default_cwd),
            title=title,
            discord_title=title,
            guild_id=getattr(source, "guild_id", None),
            parent_chat_id=getattr(source, "parent_chat_id", None),
        )
        self.bindings.bind(binding)
        self._gateway_started_thread_ids.discard(thread_id)
        with suppress(Exception):
            await self.client.request(
                "thread/name/set", {"threadId": thread_id, "name": title}
            )
        await self._rename_bound_task(thread_id, title, cwd=binding.cwd)
        return binding

    async def _new_from_command(self, event: Any, cwd_arg: str) -> str:
        """Replace this Discord thread's mapping with a fresh Codex task."""
        cwd = Path(cwd_arg).expanduser() if cwd_arg else self.settings.default_cwd
        if not cwd.is_absolute():
            return "Usage: `/new /absolute/working/directory`"
        result = await self._start_gateway_thread(
            {"cwd": str(cwd), "historyMode": "paginated"}
        )
        thread = result.get("thread", {}) if isinstance(result, dict) else {}
        thread_id = str(thread.get("id") or "")
        if not thread_id:
            raise RuntimeError("Codex thread/start returned no task id")
        source = event.source
        chat_id = str(source.chat_id)
        previous_binding = self.bindings.bindings.get(chat_id)
        source_title = str(
            getattr(source, "chat_name", None) or "Discord Codex task"
        )
        hermes_owned_title = (
            (previous_binding.discord_title or previous_binding.title)
            if previous_binding is not None
            else ""
        )
        title = (
            previous_binding.title
            if previous_binding is not None
            and previous_binding.title
            and source_title == hermes_owned_title
            else source_title
        )
        current_discord_title = hermes_owned_title or source_title
        with suppress(Exception):
            await self.client.request(
                "thread/name/set", {"threadId": thread_id, "name": title}
            )
        binding = CodexTaskBinding(
            discord_chat_id=chat_id,
            codex_thread_id=thread_id,
            cwd=str(result.get("cwd") or cwd),
            title=title,
            discord_title=current_discord_title,
            guild_id=getattr(source, "guild_id", None),
            parent_chat_id=getattr(source, "parent_chat_id", None),
        )
        self.bindings.bind(binding)
        await self._rename_bound_task(thread_id, title, cwd=binding.cwd)
        self._gateway_started_thread_ids.discard(thread_id)
        return f"Created and attached Codex task `{thread_id}` in `{cwd}`."

    async def _start_gateway_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start a Discord-originated task without auto-creating a second thread."""
        self._pending_gateway_thread_starts += 1
        try:
            result = await self.client.request("thread/start", params)
            thread = result.get("thread", {}) if isinstance(result, dict) else {}
            thread_id = str(thread.get("id") or "")
            if thread_id:
                self._gateway_started_thread_ids.add(thread_id)
            return result
        finally:
            self._pending_gateway_thread_starts -= 1
            if self._pending_gateway_thread_starts == 0:
                await self._drain_deferred_started_threads()

    async def _resume_from_command(self, event: Any, selector: str) -> str:
        """Resolve a task id or unique title search and bind this Discord thread."""
        if not selector:
            return await self._list_tasks("")
        if _UUID_PATTERN.match(selector):
            thread_id = selector
            title = ""
        else:
            result = await self.client.request(
                "thread/list",
                {
                    "limit": 10,
                    "searchTerm": selector,
                    "sortKey": "recency_at",
                    "sortDirection": "desc",
                    "sourceKinds": ["cli", "vscode", "appServer"],
                    "useStateDbOnly": True,
                },
            )
            tasks = result.get("data", []) if isinstance(result, dict) else []
            if len(tasks) != 1:
                if not tasks:
                    return f"No Codex task title matched `{selector}`."
                return self._format_task_candidates(tasks)
            thread_id = str(tasks[0].get("id") or "")
            title = self._task_title(tasks[0])
        source = event.source
        previous = self.bindings.bindings.get(str(source.chat_id))
        current_discord_title = str(getattr(source, "chat_name", None) or "")
        if previous is not None:
            current_discord_title = (
                previous.discord_title or previous.title or current_discord_title
            )
        provisional = CodexTaskBinding(
            discord_chat_id=str(source.chat_id),
            codex_thread_id=thread_id,
            cwd=str(self.settings.default_cwd),
            title=title,
            discord_title=current_discord_title,
            guild_id=getattr(source, "guild_id", None),
            parent_chat_id=getattr(source, "parent_chat_id", None),
        )
        displaced = [
            stored
            for chat_id, stored in self.bindings.bindings.items()
            if chat_id != provisional.discord_chat_id
            and stored.codex_thread_id == provisional.codex_thread_id
        ]
        self.bindings.bind(provisional)
        try:
            result = await self._resume_and_sync_binding(provisional)
        except Exception:
            self.bindings.unbind(provisional.discord_chat_id)
            if previous is not None:
                self.bindings.bind(previous)
            for binding in displaced:
                self.bindings.bind(binding)
            raise
        provisional = await self._synchronize_binding_metadata(provisional, result)
        return (
            f"Attached to Codex task `{thread_id}`"
            + (f" — **{provisional.title}**" if provisional.title else "")
            + "."
        )

    async def _list_tasks(self, search_term: str) -> str:
        """List recent daemon tasks, optionally filtered by title substring."""
        params: dict[str, Any] = {
            "limit": 20,
            "sortKey": "recency_at",
            "sortDirection": "desc",
            "sourceKinds": ["cli", "vscode", "appServer"],
            "useStateDbOnly": True,
        }
        if search_term:
            params["searchTerm"] = search_term
        result = await self.client.request("thread/list", params)
        tasks = result.get("data", []) if isinstance(result, dict) else []
        if not tasks:
            return "No Codex tasks matched."
        return self._format_task_candidates(tasks)

    async def _status_text(self, chat_id: str) -> str:
        """Return the mapping and live turn known for this Discord thread."""
        binding = self.bindings.bindings.get(chat_id)
        if binding is None:
            return "This Discord thread is not attached to a Codex task."
        active_turn = self.active_turns.get(binding.codex_thread_id)
        state = f"working on turn `{active_turn}`" if active_turn else "idle"
        title = f"\nTitle: **{binding.title}**" if binding.title else ""
        return (
            f"Codex task: `{binding.codex_thread_id}`{title}\n"
            f"Directory: `{binding.cwd}`\nState: {state}"
        )

    async def _refresh_task(self, chat_id: str) -> str:
        """Resubscribe one mapped task and replay its missing durable history."""
        binding = self.bindings.bindings.get(chat_id)
        if binding is None:
            return "This Discord thread is not attached to a Codex task."
        try:
            result = await self._resume_and_sync_binding(binding)
        except CodexRpcError as error:
            if _is_missing_rollout_error(error, binding.codex_thread_id):
                return (
                    "Codex has not written this task's rollout yet. Try `/refresh` "
                    "again after the first prompt starts."
                )
            return f"Could not refresh Codex task `{binding.codex_thread_id}`: {error.message}"
        binding = await self._synchronize_binding_metadata(binding, result)
        status = await self._status_text(chat_id)
        return f"Refreshed Codex task `{binding.codex_thread_id}`.\n{status}"

    async def _synchronize_binding_metadata(
        self,
        binding: CodexTaskBinding,
        resume_result: dict[str, Any],
    ) -> CodexTaskBinding:
        """Align one binding and its Discord title with resumed Codex metadata."""
        thread = (
            resume_result.get("thread", {})
            if isinstance(resume_result, dict)
            else {}
        )
        title = binding.title
        if isinstance(thread, dict) and (thread.get("name") or thread.get("preview")):
            title = self._task_title(thread)
        cwd = str(
            resume_result.get("cwd")
            or (thread.get("cwd") if isinstance(thread, dict) else "")
            or binding.cwd
        )
        await self._rename_bound_task(binding.codex_thread_id, title, cwd=cwd)
        synchronized = self.bindings.bindings.get(binding.discord_chat_id, binding)
        synchronized.cwd = cwd
        self.bindings.bind(synchronized)
        return synchronized

    async def _interrupt_task(self, chat_id: str) -> str:
        """Interrupt the exact active Codex turn attached to a Discord thread."""
        binding = self.bindings.bindings.get(chat_id)
        if binding is None:
            return "This Discord thread is not attached to a Codex task."
        turn_id = self.active_turns.get(binding.codex_thread_id)
        if turn_id is None:
            await self._resume_binding(binding, show_activity=False)
            await self._update_bound_task_status(binding.codex_thread_id)
            turn_id = self.active_turns.get(binding.codex_thread_id)
        if turn_id is None:
            return "The Codex task is idle."
        await self.client.request(
            "turn/interrupt",
            {"threadId": binding.codex_thread_id, "turnId": turn_id},
        )
        return f"Interrupt requested for Codex turn `{turn_id}`."

    async def _send_event_to_codex(
        self, binding: CodexTaskBinding, event: Any
    ) -> None:
        """Start or steer a turn while preserving the Discord message identity."""
        inputs = self._codex_inputs(event)
        if not inputs:
            raise ValueError("The Discord message has no Codex-compatible content")
        client_message_id = str(event.message_id or "") or None
        if client_message_id:
            self.submitted_client_ids.add(client_message_id)
            self.bindings.set_discord_message_pending(
                binding.discord_chat_id,
                client_message_id,
                pending=True,
            )
        params = {
            "threadId": binding.codex_thread_id,
            "input": inputs,
            "clientUserMessageId": client_message_id,
        }
        turn_id = self.active_turns.get(binding.codex_thread_id)
        try:
            if turn_id:
                result = await self.client.request(
                    "turn/steer", {**params, "expectedTurnId": turn_id}
                )
            else:
                result = await self.client.request("turn/start", params)
        except CodexRpcError:
            await self._resume_binding(binding, show_activity=False)
            turn_id = self.active_turns.get(binding.codex_thread_id)
            if turn_id:
                result = await self.client.request(
                    "turn/steer", {**params, "expectedTurnId": turn_id}
                )
            else:
                result = await self.client.request("turn/start", params)
        turn = result.get("turn", {}) if isinstance(result, dict) else {}
        returned_turn_id = str(turn.get("id") or "")
        if returned_turn_id:
            self.active_turns[binding.codex_thread_id] = returned_turn_id
            await self._update_bound_task_status(binding.codex_thread_id)

    def _codex_inputs(self, event: Any) -> list[dict[str, Any]]:
        """Convert Discord text and cached media paths into Codex user inputs."""
        inputs: list[dict[str, Any]] = []
        text = str(event.text or "").strip()
        if text:
            inputs.append({"type": "text", "text": text})
        for path, media_type in zip(event.media_urls or [], event.media_types or []):
            if str(media_type).startswith("image/"):
                inputs.append({"type": "localImage", "path": str(path)})
            elif str(media_type).startswith("audio/"):
                inputs.append({"type": "localAudio", "path": str(path)})
        return inputs

    async def _resume_bound_tasks(self) -> None:
        """Rejoin, synchronize, and discover every daemon-owned Codex task."""
        await self._clear_socket_scoped_state()
        self.active_turns.clear()
        bindings = list(self.bindings.bindings.values())
        self._history_syncing_thread_ids.update(
            binding.codex_thread_id for binding in bindings
        )
        for binding in bindings:
            try:
                result = await self._resume_and_sync_binding(binding)
                await self._synchronize_binding_metadata(binding, result)
            except Exception:
                logger.exception(
                    "Could not synchronize mapped Codex task %s",
                    binding.codex_thread_id,
                )
        await self._discover_loaded_tasks()

    async def _discover_loaded_tasks(self) -> None:
        """Create Discord threads for every loaded task returned by the daemon."""
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {"limit": _HISTORY_PAGE_SIZE}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                result = await self.client.request("thread/loaded/list", params)
            except Exception:
                logger.exception("Could not list loaded Codex tasks for Discord discovery")
                return
            thread_ids = result.get("data", []) if isinstance(result, dict) else []
            for raw_thread_id in thread_ids:
                await self._discover_loaded_task(str(raw_thread_id or ""))
            cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not cursor:
                return

    async def _discover_loaded_task(self, thread_id: str) -> None:
        """Create a Discord mapping for one loaded task that lacks one."""
        if not thread_id or self.bindings.chats_for_codex_thread(thread_id):
            return
        try:
            read_result = await self.client.request(
                "thread/read", {"threadId": thread_id, "includeTurns": False}
            )
            thread = (
                read_result.get("thread", {})
                if isinstance(read_result, dict)
                else {}
            )
            await self._handle_started_thread(thread)
        except Exception:
            logger.exception("Could not discover loaded Codex task %s", thread_id)

    async def _clear_socket_scoped_state(self) -> None:
        """Discard request and stream identities that belonged to a replaced socket."""
        await self._cancel_all_provisional_threads()
        tasks = [
            stream.flush_task
            for stream in self.stream_messages.values()
            if stream.flush_task is not None and not stream.flush_task.done()
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self.pending_server_requests.clear()
        self.stream_messages.clear()
        self.turn_displays.clear()
        self.item_phases.clear()
        self.submitted_client_ids = {
            message_id
            for binding in self.bindings.bindings.values()
            for message_id in binding.pending_discord_message_ids
        }
        self._history_syncing_thread_ids.clear()
        self._queued_history_notifications.clear()

    async def _resume_and_sync_binding(
        self, binding: CodexTaskBinding
    ) -> dict[str, Any]:
        """Subscribe to one task and fill every durable Discord history gap."""
        thread_id = binding.codex_thread_id
        self._history_syncing_thread_ids.add(thread_id)
        try:
            result = await self._resume_binding(binding)
            await self._sync_task_history(binding)
            return result
        finally:
            await self._finish_history_sync(thread_id)

    async def _sync_task_history(self, binding: CodexTaskBinding) -> None:
        """Replay every undelivered durable item in chronological turn order."""
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {
                "threadId": binding.codex_thread_id,
                "limit": _HISTORY_PAGE_SIZE,
                "sortDirection": "asc",
                "itemsView": "full",
            }
            if cursor is not None:
                params["cursor"] = cursor
            result = await self.client.request("thread/turns/list", params)
            turns = result.get("data", []) if isinstance(result, dict) else []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                turn_id = str(turn.get("id") or "")
                turn_is_final = turn.get("status") != "inProgress"
                item_params = {
                    "threadId": binding.codex_thread_id,
                    "turnId": turn_id,
                }
                for item in turn.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_status = str(item.get("status") or "")
                    item_is_final = item_status not in {"inProgress", "running"}
                    await self._show_item(
                        binding.discord_chat_id,
                        item_params,
                        item,
                        final=turn_is_final or item_is_final,
                    )
                await self._register_history_turn_activity(
                    binding.discord_chat_id,
                    turn_id,
                    status=str(turn.get("status") or "completed"),
                )
            cursor = result.get("nextCursor") if isinstance(result, dict) else None
            if not cursor:
                return

    async def _finish_history_sync(self, thread_id: str) -> None:
        """Resume live rendering after one task's durable history is synchronized."""
        self._history_syncing_thread_ids.discard(thread_id)
        active_turn_id = self.active_turns.get(thread_id)
        if active_turn_id:
            for chat_id in self.bindings.chats_for_codex_thread(thread_id):
                await self._show_turn_started(chat_id, active_turn_id)
        notifications = self._queued_history_notifications.pop(thread_id, [])
        for method, params in notifications:
            await self.handle_notification(method, params)

    async def _register_history_turn_activity(
        self, chat_id: str, turn_id: str, *, status: str
    ) -> None:
        """Restore activity buttons or finalize a card created by history replay."""
        key = (chat_id, turn_id)
        display = self.turn_displays.get(key)
        if display is None:
            return
        display.status = status
        if display.rendered_on_connection:
            await self._update_turn_activity_locked(chat_id, turn_id, display)
        elif (
            display.status_message_id
            and display.details
            and not display.legacy_split_messages
        ):
            register = getattr(self.adapter, "register_codex_activity", None)
            if register is not None:
                register(
                    display.status_message_id,
                    self._collapsed_turn_activity(display),
                    self._expanded_turn_activity(display),
                )
        if status != "inProgress":
            self.turn_displays.pop(key, None)

    async def _resume_binding(
        self,
        binding: CodexTaskBinding,
        *,
        show_activity: bool = True,
    ) -> dict[str, Any]:
        """Subscribe to one task and recover its active turn identity."""
        result = await self.client.request(
            "thread/resume",
            {
                "threadId": binding.codex_thread_id,
                "excludeTurns": True,
                "initialTurnsPage": {
                    "limit": 1,
                    "sortDirection": "desc",
                    "itemsView": "full",
                },
            },
        )
        active_turn_id = self._active_turn_id(result)
        if active_turn_id:
            self.active_turns[binding.codex_thread_id] = active_turn_id
            if (
                show_activity
                and binding.codex_thread_id not in self._history_syncing_thread_ids
            ):
                for chat_id in self.bindings.chats_for_codex_thread(
                    binding.codex_thread_id
                ):
                    await self._show_turn_started(chat_id, active_turn_id)
        else:
            self.active_turns.pop(binding.codex_thread_id, None)
        return result

    def _active_turn_id(self, result: Any) -> Optional[str]:
        """Extract an in-progress turn id from a paged resume response."""
        if not isinstance(result, dict):
            return None
        page = result.get("initialTurnsPage")
        turns = page.get("data", []) if isinstance(page, dict) else []
        for turn in turns:
            if isinstance(turn, dict) and turn.get("status") == "inProgress":
                turn_id = str(turn.get("id") or "")
                return turn_id or None
        return None

    async def handle_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        """Mirror live Codex lifecycle and item notifications into Discord."""
        if method == "serverRequest/resolved":
            self._remove_resolved_request(params.get("requestId"))
            return
        if method == "thread/started":
            await self._handle_started_thread(params.get("thread") or {})
            return
        thread_id = str(
            params.get("threadId")
            or (params.get("thread") or {}).get("id")
            or ""
        )
        if not thread_id:
            return
        if method == "thread/closed":
            self._deferred_started_threads.pop(thread_id, None)
            await self._cancel_provisional_thread(thread_id)
            self.active_turns.pop(thread_id, None)
            await self._update_bound_task_status(thread_id)
            return
        if thread_id in self._history_syncing_thread_ids:
            self._queued_history_notifications.setdefault(thread_id, []).append(
                (method, params)
            )
            return
        chat_ids = self.bindings.chats_for_codex_thread(thread_id)
        if not chat_ids:
            return
        if method == "thread/name/updated":
            await self._rename_bound_task(thread_id, str(params.get("threadName") or ""))
            return
        if method == "turn/started":
            turn = params.get("turn") or {}
            turn_id = str(turn.get("id") or "")
            if turn_id:
                self.active_turns[thread_id] = turn_id
                await self._update_bound_task_status(thread_id)
                for chat_id in chat_ids:
                    await self._show_turn_started(chat_id, turn_id)
            return
        if method == "turn/completed":
            turn = params.get("turn") or {}
            turn_id = str(turn.get("id") or "")
            if self.active_turns.get(thread_id) == turn_id:
                self.active_turns.pop(thread_id, None)
            await self._update_bound_task_status(thread_id)
            for chat_id in chat_ids:
                await self._show_turn_completed(chat_id, turn)
            return
        if method in {"item/agentMessage/delta", "item/plan/delta"}:
            item_id = str(params.get("itemId") or "")
            prefix = "**Plan**\n" if method == "item/plan/delta" else ""
            for chat_id in chat_ids:
                activity_kind = None
                if method == "item/plan/delta":
                    activity_kind = "plan"
                elif self.item_phases.get((chat_id, item_id)) == "commentary":
                    activity_kind = "update"
                    prefix = "**Update**\n"
                await self._append_stream_delta(
                    chat_id,
                    str(params.get("turnId") or ""),
                    item_id,
                    str(params.get("delta") or ""),
                    prefix,
                    activity_kind=activity_kind,
                )
            return
        if method == "item/commandExecution/outputDelta":
            for chat_id in chat_ids:
                await self._append_stream_delta(
                    chat_id,
                    str(params.get("turnId") or ""),
                    str(params.get("itemId") or ""),
                    str(params.get("delta") or ""),
                    "",
                    activity_kind="command",
                )
            return
        if method == "item/reasoning/summaryPartAdded":
            if int(params.get("summaryIndex") or 0) > 0:
                for chat_id in chat_ids:
                    await self._append_stream_delta(
                        chat_id,
                        str(params.get("turnId") or ""),
                        str(params.get("itemId") or ""),
                        "\n\n",
                        "**Reasoning**\n",
                        activity_kind="reasoning",
                    )
            return
        if method == "item/reasoning/summaryTextDelta":
            for chat_id in chat_ids:
                await self._append_stream_delta(
                    chat_id,
                    str(params.get("turnId") or ""),
                    str(params.get("itemId") or ""),
                    str(params.get("delta") or ""),
                    "**Reasoning**\n",
                    activity_kind="reasoning",
                )
            return
        if method == "item/started":
            item = params.get("item") or {}
            for chat_id in chat_ids:
                item_id = str(item.get("id") or "")
                if item_id and item.get("type") == "agentMessage":
                    self.item_phases[(chat_id, item_id)] = str(
                        item.get("phase") or ""
                    )
                await self._show_item(chat_id, params, item, final=False)
            return
        if method == "item/completed":
            item = params.get("item") or {}
            if item.get("type") == "userMessage":
                await self._rename_untitled_task(
                    thread_id, self._user_message_text(item)
                )
            for chat_id in chat_ids:
                await self._show_item(chat_id, params, item, final=True)
            return
        if method == "error":
            error = params.get("error") or {}
            message = str(error.get("message") or "Codex turn failed")
            retry = " Codex will retry." if params.get("willRetry") else ""
            for chat_id in chat_ids:
                await self.adapter.send(chat_id, f"⚠️ **Codex error:** {message}{retry}")

    async def _handle_started_thread(self, thread: dict[str, Any]) -> None:
        """Start one provisional mirror task for an external Codex task."""
        thread_id = str(thread.get("id") or "")
        if not thread_id or self.bindings.chats_for_codex_thread(thread_id):
            return
        if thread_id in self._gateway_started_thread_ids:
            return
        if self._pending_gateway_thread_starts:
            self._deferred_started_threads[thread_id] = thread
            return
        if thread_id in self._provisional_thread_tasks:
            return
        first_attempt_finished = asyncio.Event()
        self._awaiting_rollout_thread_ids.add(thread_id)
        task = asyncio.create_task(
            self._mirror_external_thread(thread, first_attempt_finished),
            name=f"codex-rollout-{thread_id}",
        )
        self._provisional_thread_tasks[thread_id] = task
        await first_attempt_finished.wait()

    async def _drain_deferred_started_threads(self) -> None:
        """Process external task notifications after local starts have identities."""
        deferred = list(self._deferred_started_threads.values())
        self._deferred_started_threads.clear()
        for thread in deferred:
            thread_id = str(thread.get("id") or "")
            if thread_id in self._gateway_started_thread_ids:
                continue
            await self._handle_started_thread(thread)

    async def _mirror_external_thread(
        self,
        thread: dict[str, Any],
        first_attempt_finished: asyncio.Event,
    ) -> None:
        """Wait for rollout durability, then create and synchronize Discord."""
        thread_id = str(thread.get("id") or "")
        provisional = CodexTaskBinding(
            discord_chat_id="",
            codex_thread_id=thread_id,
            cwd=str(thread.get("cwd") or self.settings.default_cwd),
            title=self._task_title(thread),
            parent_chat_id=self.settings.parent_chat_id,
        )
        retry_delay = _ROLLOUT_RETRY_INITIAL_DELAY_SECONDS
        binding: Optional[CodexTaskBinding] = None
        self._history_syncing_thread_ids.add(thread_id)
        try:
            while True:
                try:
                    resume_result = await self._resume_binding(provisional)
                except CodexRpcError as error:
                    if not _is_missing_rollout_error(error, thread_id):
                        raise
                    if not first_attempt_finished.is_set():
                        logger.info(
                            "Waiting for Codex task %s to write its rollout",
                            thread_id,
                        )
                        first_attempt_finished.set()
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(
                        max(retry_delay * 2, _ROLLOUT_RETRY_INITIAL_DELAY_SECONDS),
                        _ROLLOUT_RETRY_MAX_DELAY_SECONDS,
                    )
                    continue
                self._awaiting_rollout_thread_ids.discard(thread_id)
                resumed_thread = (
                    resume_result.get("thread", {})
                    if isinstance(resume_result, dict)
                    else {}
                )
                # Resume metadata reflects the durable rollout; the start event
                # supplies only fields omitted from the resume response.
                synchronized_thread = dict(thread)
                if isinstance(resumed_thread, dict):
                    for field_name in ("name", "preview"):
                        if field_name in resumed_thread:
                            synchronized_thread[field_name] = resumed_thread.get(
                                field_name
                            )
                synchronized_thread["id"] = thread_id
                synchronized_thread["cwd"] = str(
                    resume_result.get("cwd")
                    or (
                        resumed_thread.get("cwd")
                        if isinstance(resumed_thread, dict)
                        else ""
                    )
                    or thread.get("cwd")
                    or self.settings.default_cwd
                )
                binding = await self._create_discord_thread_for_task(
                    synchronized_thread
                )
                if binding is not None:
                    await self._sync_task_history(binding)
                    logger.info(
                        "Synchronized Codex task %s after rollout became available",
                        thread_id,
                    )
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Could not mirror external Codex task %s", thread_id)
        finally:
            try:
                if binding is not None:
                    await self._finish_history_sync(thread_id)
                else:
                    self._discard_provisional_thread_state(thread_id)
            except Exception:
                self._history_syncing_thread_ids.discard(thread_id)
                self._queued_history_notifications.pop(thread_id, None)
                logger.exception(
                    "Could not finish Discord history synchronization for Codex task %s",
                    thread_id,
                )
            finally:
                if not first_attempt_finished.is_set():
                    first_attempt_finished.set()
                current_task = asyncio.current_task()
                if self._provisional_thread_tasks.get(thread_id) is current_task:
                    self._provisional_thread_tasks.pop(thread_id, None)
                self._awaiting_rollout_thread_ids.discard(thread_id)

    async def _cancel_provisional_thread(self, thread_id: str) -> None:
        """Cancel one provisional mirror after Codex closes the task."""
        if thread_id not in self._awaiting_rollout_thread_ids:
            return
        task = self._provisional_thread_tasks.get(thread_id)
        if task is None:
            return
        task.cancel()
        if task is not asyncio.current_task():
            with suppress(asyncio.CancelledError):
                await task

    async def _cancel_all_provisional_threads(self) -> None:
        """Cancel every rollout retry before replacing the socket or stopping."""
        tasks = list(self._provisional_thread_tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    def _discard_provisional_thread_state(self, thread_id: str) -> None:
        """Discard queued events for a task that never gained a Discord mapping."""
        self._history_syncing_thread_ids.discard(thread_id)
        self._queued_history_notifications.pop(thread_id, None)
        self.active_turns.pop(thread_id, None)
        self._awaiting_rollout_thread_ids.discard(thread_id)

    async def _create_discord_thread_for_task(
        self, thread: dict[str, Any]
    ) -> Optional[CodexTaskBinding]:
        """Create and persist Discord after an external task becomes resumable."""
        thread_id = str(thread.get("id") or "")
        existing_chat_ids = self.bindings.chats_for_codex_thread(thread_id)
        if existing_chat_ids:
            return self.bindings.bindings.get(existing_chat_ids[0])
        parent_chat_id = self.settings.parent_chat_id
        if not thread_id or not parent_chat_id:
            logger.warning(
                "Cannot create a Discord thread for Codex task %s without parent_chat_id",
                thread_id,
            )
            return None
        title = self._task_title(thread)
        cwd = str(thread.get("cwd") or self.settings.default_cwd)
        discord_title = _discord_thread_title(
            title,
            cwd,
            thread_id,
            working=thread_id in self.active_turns,
        )
        discord_chat_id = await self.adapter.create_codex_task_thread(
            parent_chat_id,
            discord_title,
            member_user_ids=self.settings.member_user_ids,
        )
        if not discord_chat_id:
            logger.error("Could not create a Discord thread for Codex task %s", thread_id)
            return None
        binding = CodexTaskBinding(
            discord_chat_id=str(discord_chat_id),
            codex_thread_id=thread_id,
            cwd=cwd,
            title=title,
            discord_title=discord_title,
            parent_chat_id=parent_chat_id,
        )
        self.bindings.bind(binding)
        return binding

    async def _rename_untitled_task(self, thread_id: str, prompt: str) -> None:
        """Use the first terminal prompt as the title of an unnamed Codex task."""
        if not any(
            binding.title == "Untitled task"
            for binding in self.bindings.bindings.values()
            if binding.codex_thread_id == thread_id
        ):
            return
        title = prompt.strip().splitlines()[0][:80]
        if title:
            await self._rename_bound_task(thread_id, title)

    async def _update_bound_task_status(self, thread_id: str) -> None:
        """Render one task's active or idle status in its Discord title."""
        chat_ids = self.bindings.chats_for_codex_thread(thread_id)
        if not chat_ids:
            return
        binding = self.bindings.bindings.get(chat_ids[0])
        if binding is not None:
            await self._rename_bound_task(
                thread_id,
                binding.title or "Codex task",
            )

    async def _rename_bound_task(
        self,
        thread_id: str,
        title: str,
        *,
        cwd: Optional[str] = None,
    ) -> None:
        """Render Codex metadata into mapped Discord thread titles."""
        cleaned_title = re.sub(r"\s+", " ", title).strip()[:80]
        if not cleaned_title:
            return
        for chat_id in self.bindings.chats_for_codex_thread(thread_id):
            binding = self.bindings.bindings.get(chat_id)
            if binding is None:
                continue
            expected_discord_title = (
                binding.discord_title or (binding.title or "Codex task").strip()[:80]
            )
            desired_discord_title = _discord_thread_title(
                cleaned_title,
                cwd or binding.cwd,
                binding.codex_thread_id,
                working=thread_id in self.active_turns,
            )
            if desired_discord_title == expected_discord_title:
                binding.title = cleaned_title
                binding.discord_title = expected_discord_title
                self.bindings.bind(binding)
                continue
            renamed = await self.adapter.rename_thread(
                chat_id,
                desired_discord_title,
                only_if_current_name=expected_discord_title,
            )
            binding.title = cleaned_title
            binding.discord_title = (
                desired_discord_title if renamed else expected_discord_title
            )
            self.bindings.bind(binding)

    async def handle_server_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        """Forward approvals and user-input requests to the mapped Discord thread."""
        if method == "currentTime/read":
            await self.client.respond(
                request_id, {"currentTimeAt": int(datetime.now(timezone.utc).timestamp())}
            )
            return
        thread_id = str(params.get("threadId") or params.get("conversationId") or "")
        chat_ids = self.bindings.chats_for_codex_thread(thread_id)
        if not chat_ids:
            await self.client.reject(
                request_id, -32601, "No Discord thread is attached to this Codex task"
            )
            return
        supported = {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
            "item/tool/requestUserInput",
            "mcpServer/elicitation/request",
        }
        if method not in supported:
            for chat_id in chat_ids:
                await self.adapter.send(
                    chat_id,
                    f"⚠️ Codex requested unsupported client action `{method}`. "
                    "Continue from a Codex terminal client for this request.",
                )
            await self.client.reject(
                request_id, -32601, f"Discord gateway does not implement {method}"
            )
            return
        for chat_id in chat_ids:
            queue = self.pending_server_requests.setdefault(chat_id, [])
            queue.append(PendingCodexServerRequest(request_id, method, params))
            await self.adapter.send(chat_id, self._server_request_prompt(method, params))

    async def _show_turn_started(self, chat_id: str, turn_id: str) -> None:
        """Create one collapsed activity message for a newly observed turn."""
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            display = self.turn_displays.setdefault(
                (chat_id, turn_id), DiscordTurnDisplay()
            )
            if display.status_message_id:
                return
            await self._update_turn_activity_locked(chat_id, turn_id, display)

    async def _show_turn_completed(
        self, chat_id: str, turn: dict[str, Any]
    ) -> None:
        """Finalize the collapsed activity message without moving the answer."""
        turn_id = str(turn.get("id") or "")
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            display = self.turn_displays.get((chat_id, turn_id))
            if display is None:
                return
            display.status = str(turn.get("status") or "completed")
            await self._update_turn_activity_locked(chat_id, turn_id, display)
            self.turn_displays.pop((chat_id, turn_id), None)

    async def _set_turn_activity_detail_locked(
        self,
        chat_id: str,
        turn_id: str,
        item_id: str,
        text: str,
        *,
        kind: str,
        failed: bool = False,
        final: bool = False,
    ) -> Optional[str]:
        """Update one detail inside the turn-owned Discord activity message."""
        display = self.turn_displays.setdefault(
            (chat_id, turn_id), DiscordTurnDisplay()
        )
        display.details[item_id] = DiscordActivityDetail(
            text=text,
            kind=kind,
            failed=failed,
        )
        await self._update_turn_activity_locked(chat_id, turn_id, display)
        message_id = display.status_message_id
        if final and message_id:
            self.bindings.record_item_delivery(
                chat_id,
                item_id,
                CodexItemDelivery(
                    discord_message_id=message_id,
                    final=True,
                ),
            )
        return message_id

    async def _update_turn_activity_locked(
        self,
        chat_id: str,
        turn_id: str,
        display: DiscordTurnDisplay,
    ) -> None:
        """Send or edit the one activity message that belongs to a Codex turn."""
        collapsed = self._collapsed_turn_activity(display)
        expanded = self._expanded_turn_activity(display)
        metadata = {
            "message_style": "codex_turn_activity",
            "expanded_content": expanded,
        }
        if display.status_message_id:
            result = await self.adapter.edit_message(
                chat_id,
                display.status_message_id,
                collapsed,
                finalize=display.status != "inProgress",
                metadata=metadata,
            )
        else:
            result = await self.adapter.send(
                chat_id,
                collapsed,
                metadata=metadata,
            )
        if result.success and result.message_id:
            display.status_message_id = str(result.message_id)
            display.rendered_on_connection = True

    def _collapsed_turn_activity(self, display: DiscordTurnDisplay) -> str:
        """Summarize a turn's activity counts for the default collapsed view."""
        if not display.details:
            if display.status == "inProgress":
                return "⏳ **Codex is working…**"
            return f"✅ **Codex turn {display.status}.**"
        counts: dict[str, int] = {}
        failed_count = 0
        for detail in display.details.values():
            counts[detail.kind] = counts.get(detail.kind, 0) + 1
            failed_count += int(detail.failed)
        labels = []
        for kind in ("command", "update", "reasoning", "plan", "tool"):
            count = counts.get(kind, 0)
            if count:
                noun = kind if count == 1 else f"{kind}s"
                labels.append(f"{count} {noun}")
        if failed_count:
            labels.append(f"{failed_count} failed")
        icon = "⏳" if display.status == "inProgress" else "✅"
        total = len(display.details)
        step_label = "step" if total == 1 else "steps"
        return f"{icon} **Activity · {total} {step_label}**\n" + " · ".join(labels)

    def _expanded_turn_activity(self, display: DiscordTurnDisplay) -> str:
        """Render every recorded activity detail for Show activity pagination."""
        if not display.details:
            return self._collapsed_turn_activity(display)
        status = "working" if display.status == "inProgress" else display.status
        details = "\n\n---\n\n".join(
            detail.text for detail in display.details.values()
        )
        return f"### Codex activity · {status}\n\n{details}"

    async def _append_stream_delta(
        self,
        chat_id: str,
        turn_id: str,
        item_id: str,
        delta: str,
        prefix: str,
        *,
        activity_kind: Optional[str] = None,
    ) -> None:
        """Accumulate one item delta and schedule a rate-limited Discord edit."""
        if not item_id or not delta:
            return
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            delivery = self.bindings.item_delivery(chat_id, item_id)
            if delivery is not None and delivery.final:
                return
            stream = self._item_stream(
                chat_id,
                turn_id,
                item_id,
                prefix=prefix,
                activity_kind=activity_kind,
            )
            stream.text += delta
            if stream.flush_task is None or stream.flush_task.done():
                stream.flush_task = asyncio.create_task(
                    self._flush_stream_message(chat_id, item_id),
                    name=f"codex-discord-stream-{item_id}",
                )

    async def _flush_stream_message(self, chat_id: str, item_id: str) -> None:
        """Send or edit the latest cumulative text for one streaming Codex item."""
        await asyncio.sleep(_STREAM_EDIT_INTERVAL_SECONDS)
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            stream = self.stream_messages.get((chat_id, item_id))
            if stream is None:
                return
            content = stream.prefix + stream.text + stream.suffix
            if stream.activity_kind:
                stream.message_id = await self._set_turn_activity_detail_locked(
                    chat_id,
                    stream.turn_id,
                    item_id,
                    content,
                    kind=stream.activity_kind,
                )
                result = None
            elif stream.message_id:
                result = await self.adapter.edit_message(
                    chat_id, stream.message_id, content
                )
            else:
                result = await self.adapter.send(chat_id, content)
            if result is not None and result.success and result.message_id:
                stream.message_id = str(result.message_id)
            if stream.message_id:
                self._record_stream_delivery(chat_id, item_id, stream, final=False)

    async def _show_item(
        self,
        chat_id: str,
        params: dict[str, Any],
        item: dict[str, Any],
        *,
        final: bool,
    ) -> None:
        """Render an authoritative completed item or a visible activity start."""
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            await self._show_item_locked(chat_id, params, item, final=final)

    async def _show_item_locked(
        self,
        chat_id: str,
        params: dict[str, Any],
        item: dict[str, Any],
        *,
        final: bool,
    ) -> None:
        """Render one Codex item while its Discord thread delivery lock is held."""
        item_type = str(item.get("type") or "")
        item_id = str(item.get("id") or "")
        turn_id = str(params.get("turnId") or "")
        if not item_id:
            return
        if item_type == "agentMessage":
            self.item_phases[(chat_id, item_id)] = str(item.get("phase") or "")
        activity_detail = self._activity_detail_for_item(item, final=final)
        delivery = self.bindings.item_delivery(chat_id, item_id)
        if delivery is not None and delivery.final:
            if activity_detail is not None:
                self._remember_delivered_activity(
                    chat_id,
                    turn_id,
                    item_id,
                    delivery,
                    activity_detail,
                )
            return
        if item_type == "userMessage":
            if not final:
                return
            client_id = str(item.get("clientId") or "")
            if client_id and client_id in self.submitted_client_ids:
                self.submitted_client_ids.discard(client_id)
                self.bindings.set_discord_message_pending(
                    chat_id,
                    client_id,
                    pending=False,
                )
                self.bindings.record_item_delivery(
                    chat_id,
                    item_id,
                    CodexItemDelivery(discord_message_id=client_id, final=True),
                )
                return
            text = self._user_message_text(item)
            if text:
                message_id = await self._show_external_user_message(
                    chat_id,
                    turn_id,
                    text,
                )
                if message_id:
                    self.bindings.record_item_delivery(
                        chat_id,
                        item_id,
                        CodexItemDelivery(
                            discord_message_id=message_id,
                            final=True,
                        ),
                    )
            return
        if activity_detail is not None:
            kind, text, failed = activity_detail
            stream = self.stream_messages.get((chat_id, item_id))
            if final and stream and stream.flush_task and not stream.flush_task.done():
                stream.flush_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream.flush_task
            message_id = await self._set_turn_activity_detail_locked(
                chat_id,
                turn_id,
                item_id,
                text,
                kind=kind,
                failed=failed,
                final=final,
            )
            if item_type == "commandExecution" and not final:
                stream = self._item_stream(
                    chat_id,
                    turn_id,
                    item_id,
                    activity_kind="command",
                )
                stream.message_id = message_id
                stream.prefix = text + "\n```text\n"
                stream.suffix = "\n```"
                self._record_stream_delivery(
                    chat_id, item_id, stream, final=False
                )
            elif final:
                self.stream_messages.pop((chat_id, item_id), None)
            return
        if item_type == "agentMessage":
            if not final:
                return
            await self._finalize_text_item(
                chat_id,
                turn_id,
                item_id,
                str(item.get("text") or ""),
            )
            return

    def _activity_detail_for_item(
        self, item: dict[str, Any], *, final: bool
    ) -> Optional[tuple[str, str, bool]]:
        """Classify one Codex item as collapsible activity and render its text."""
        item_type = str(item.get("type") or "")
        if item_type == "agentMessage":
            if str(item.get("phase") or "") != "commentary" or not final:
                return None
            text = str(item.get("text") or "").strip()
            return ("update", f"**Update**\n{text}", False) if text else None
        if item_type == "plan":
            text = str(item.get("text") or "").strip()
            return ("plan", f"**Plan**\n{text}", False) if text and final else None
        if item_type == "reasoning":
            summary = [str(part) for part in item.get("summary") or [] if part]
            if not summary or not final:
                return None
            return ("reasoning", "**Reasoning**\n" + "\n\n".join(summary), False)
        rendered = self._render_activity_item(item, final=final)
        if not rendered:
            return None
        kind = "command" if item_type == "commandExecution" else "tool"
        failed = str(item.get("status") or "").lower() == "failed"
        return (kind, rendered, failed)

    def _remember_delivered_activity(
        self,
        chat_id: str,
        turn_id: str,
        item_id: str,
        delivery: CodexItemDelivery,
        detail: tuple[str, str, bool],
    ) -> None:
        """Rebuild button content for an aggregated activity message after restart."""
        kind, text, failed = detail
        display = self.turn_displays.setdefault(
            (chat_id, turn_id), DiscordTurnDisplay()
        )
        if delivery.discord_message_id:
            if display.status_message_id is None:
                display.status_message_id = delivery.discord_message_id
            elif display.status_message_id != delivery.discord_message_id:
                display.legacy_split_messages = True
        display.details[item_id] = DiscordActivityDetail(
            text=text,
            kind=kind,
            failed=failed,
        )

    async def _finalize_text_item(
        self, chat_id: str, turn_id: str, item_id: str, text: str
    ) -> Optional[str]:
        """Replace streamed text with the authoritative completed item text."""
        key = (chat_id, item_id)
        stream = self.stream_messages.pop(key, None)
        if stream and stream.flush_task and not stream.flush_task.done():
            stream.flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream.flush_task
        if stream is None:
            stream = self._item_stream(chat_id, turn_id, item_id)
        if stream.message_id:
            result = await self.adapter.edit_message(
                chat_id, stream.message_id, text, finalize=True
            )
        elif text:
            result = await self.adapter.send(chat_id, text)
        else:
            return None
        if not result.success or not result.message_id:
            return None
        message_id = str(result.message_id)
        self.stream_messages.pop(key, None)
        self.bindings.record_item_delivery(
            chat_id,
            item_id,
            CodexItemDelivery(discord_message_id=message_id, final=True),
        )
        return message_id

    async def _show_external_user_message(
        self, chat_id: str, turn_id: str, text: str
    ) -> Optional[str]:
        """Place terminal input before the assistant stream for the same turn."""
        metadata = {"message_style": "codex_user_prompt"}
        display = self.turn_displays.get((chat_id, turn_id))
        if display is None or not display.status_message_id:
            result = await self.adapter.send(chat_id, text, metadata=metadata)
            return str(result.message_id) if result.success and result.message_id else None
        result = await self.adapter.edit_message(
            chat_id,
            display.status_message_id,
            text,
            finalize=True,
            metadata=metadata,
        )
        if not result.success:
            return None
        prompt_message_id = display.status_message_id
        display.status_message_id = None
        await self._update_turn_activity_locked(chat_id, turn_id, display)
        return prompt_message_id

    def _item_stream(
        self,
        chat_id: str,
        turn_id: str,
        item_id: str,
        *,
        prefix: str = "",
        activity_kind: Optional[str] = None,
    ) -> DiscordStreamMessage:
        """Restore or create the mutable Discord stream for one Codex item."""
        key = (chat_id, item_id)
        stream = self.stream_messages.get(key)
        if stream is not None:
            return stream
        delivery = self.bindings.item_delivery(chat_id, item_id)
        stream = DiscordStreamMessage(
            turn_id=turn_id,
            activity_kind=activity_kind,
            text=delivery.text if delivery else "",
            prefix=delivery.prefix if delivery else prefix,
            suffix=delivery.suffix if delivery else "",
            message_id=delivery.discord_message_id if delivery else None,
        )
        self.stream_messages[key] = stream
        if activity_kind and stream.message_id:
            display = self.turn_displays.setdefault(
                (chat_id, turn_id), DiscordTurnDisplay()
            )
            display.status_message_id = stream.message_id
        return stream

    def _record_stream_delivery(
        self,
        chat_id: str,
        item_id: str,
        stream: DiscordStreamMessage,
        *,
        final: bool,
    ) -> None:
        """Persist enough stream state to continue the same Discord message."""
        self.bindings.record_item_delivery(
            chat_id,
            item_id,
            CodexItemDelivery(
                discord_message_id=stream.message_id,
                final=final,
                text="" if final else stream.text,
                prefix="" if final else stream.prefix,
                suffix="" if final else stream.suffix,
            ),
        )

    def _render_activity_item(
        self, item: dict[str, Any], *, final: bool
    ) -> Optional[str]:
        """Render a Codex activity item directly from its app-server fields."""
        item_type = str(item.get("type") or "")
        if item_type == "commandExecution":
            command = self._escape_code_fence(str(item.get("command") or ""))
            status = str(item.get("status") or ("completed" if final else "running"))
            output = str(item.get("aggregatedOutput") or "")
            suffix = f"\n```text\n{self._escape_code_fence(output)}\n```" if output else ""
            return f"**Command · {status}**\n```sh\n{command}\n```{suffix}"
        if item_type == "fileChange":
            changes = item.get("changes") or []
            paths = [
                str(change.get("path") or change.get("filePath") or change)
                for change in changes
            ]
            status = str(item.get("status") or ("completed" if final else "running"))
            return f"**File changes · {status}**\n" + "\n".join(
                f"- `{path}`" for path in paths
            )
        if item_type == "mcpToolCall":
            status = str(item.get("status") or ("completed" if final else "running"))
            return (
                f"**MCP tool · {status}**\n"
                f"`{item.get('server', '')}/{item.get('tool', '')}`"
            )
        if item_type == "dynamicToolCall":
            status = str(item.get("status") or ("completed" if final else "running"))
            return f"**Tool · {status}**\n`{item.get('tool', '')}`"
        if item_type == "hookPrompt":
            fragments = [
                str(fragment.get("text") or "")
                for fragment in item.get("fragments") or []
                if isinstance(fragment, dict) and fragment.get("text")
            ]
            return "**Hook prompt**\n" + "\n\n".join(fragments)
        if item_type == "collabAgentToolCall":
            status = str(item.get("status") or ("completed" if final else "running"))
            receivers = ", ".join(str(value) for value in item.get("receiverThreadIds") or [])
            receiver_line = f"\nTasks: `{receivers}`" if receivers else ""
            prompt = str(item.get("prompt") or "").strip()
            prompt_line = f"\n\n{prompt}" if prompt else ""
            return (
                f"**Agent tool · {status}**\n`{item.get('tool', '')}`"
                f"{receiver_line}{prompt_line}"
            )
        if item_type == "subAgentActivity":
            return (
                f"**Agent {item.get('kind', 'activity')}**\n"
                f"`{item.get('agentPath', '')}` · `{item.get('agentThreadId', '')}`"
            )
        if item_type == "webSearch":
            return f"**Web search**\n{item.get('query', '')}"
        if item_type == "imageView":
            return f"**Viewed image**\n`{item.get('path', '')}`"
        if item_type == "sleep":
            duration_seconds = float(item.get("durationMs") or 0) / 1000
            return f"**Sleep**\n{duration_seconds:g} seconds"
        if item_type == "imageGeneration":
            status = str(item.get("status") or ("completed" if final else "running"))
            saved_path = str(item.get("savedPath") or "")
            path_line = f"\n`{saved_path}`" if saved_path else ""
            return f"**Image generation · {status}**{path_line}"
        if item_type in {"enteredReviewMode", "exitedReviewMode"}:
            action = "Entered" if item_type == "enteredReviewMode" else "Exited"
            review = str(item.get("review") or "").strip()
            review_line = f"\n{review}" if review else ""
            return f"**{action} review mode**{review_line}"
        if item_type == "contextCompaction":
            return "**Context compacted**"
        return None

    async def _resolve_pending_request(self, chat_id: str, command: str) -> str:
        """Resolve the oldest approval request with an exact Codex result shape."""
        queue = self.pending_server_requests.get(chat_id, [])
        if not queue:
            return "No Codex approval is waiting in this Discord thread."
        pending = queue[0]
        method = pending.method
        accept = command in {"approve", "always"}
        session_scope = command == "always"
        if method == "item/commandExecution/requestApproval":
            decision = "acceptForSession" if session_scope else "accept"
            if command == "deny":
                decision = "decline"
            elif command == "cancel":
                decision = "cancel"
            result = {"decision": decision}
        elif method == "item/fileChange/requestApproval":
            decision = "acceptForSession" if session_scope else "accept"
            if command == "deny":
                decision = "decline"
            elif command == "cancel":
                decision = "cancel"
            result = {"decision": decision}
        elif method == "item/permissions/requestApproval":
            result = {
                "permissions": pending.params.get("permissions", {}) if accept else {},
                "scope": "session" if session_scope and accept else "turn",
            }
        elif method == "mcpServer/elicitation/request":
            if accept:
                if pending.params.get("mode") != "url":
                    return "Use `/answer {\"field\": \"value\"}` for this MCP form."
                result = {"action": "accept", "content": None, "_meta": None}
            else:
                action = "cancel" if command == "cancel" else "decline"
                result = {"action": action, "content": None, "_meta": None}
        else:
            return "Use `/answer ...` to answer the pending Codex question."
        queue.pop(0)
        if not queue:
            self.pending_server_requests.pop(chat_id, None)
        await self.client.respond(pending.request_id, result)
        return f"Codex request resolved with `{command}`."

    async def _answer_user_input(self, chat_id: str, answer_text: str) -> str:
        """Resolve a pending request_user_input prompt from Discord text."""
        queue = self.pending_server_requests.get(chat_id, [])
        if not queue:
            return "No Codex question is waiting in this Discord thread."
        pending = queue[0]
        if pending.method == "mcpServer/elicitation/request":
            try:
                content = json.loads(answer_text)
            except json.JSONDecodeError:
                return "Answer the MCP form with `/answer {\"field\": \"value\"}`."
            queue.pop(0)
            if not queue:
                self.pending_server_requests.pop(chat_id, None)
            await self.client.respond(
                pending.request_id,
                {"action": "accept", "content": content, "_meta": None},
            )
            return "Codex received the MCP form response."
        if pending.method != "item/tool/requestUserInput":
            return "The pending Codex request expects `/approve` or `/deny`."
        questions = pending.params.get("questions") or []
        answers: dict[str, dict[str, list[str]]] = {}
        if len(questions) == 1:
            question_id = str(questions[0].get("id") or "")
            answers[question_id] = {"answers": [answer_text.strip()]}
        else:
            parsed: dict[str, str] = {}
            for part in answer_text.split("|"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip()
            for question in questions:
                question_id = str(question.get("id") or "")
                if question_id not in parsed:
                    ids = ", ".join(str(q.get("id") or "") for q in questions)
                    return f"Answer every question as `/answer id=value | id=value`. IDs: {ids}"
                answers[question_id] = {"answers": [parsed[question_id]]}
        queue.pop(0)
        if not queue:
            self.pending_server_requests.pop(chat_id, None)
        await self.client.respond(pending.request_id, {"answers": answers})
        return "Codex received your answer."

    def _server_request_prompt(
        self, method: str, params: dict[str, Any]
    ) -> str:
        """Describe an app-server request with commands that resolve it."""
        reason = str(params.get("reason") or "").strip()
        reason_line = f"\nReason: {reason}" if reason else ""
        if method == "item/commandExecution/requestApproval":
            command = self._escape_code_fence(str(params.get("command") or ""))
            return (
                f"⚠️ **Codex requests command approval**{reason_line}\n"
                f"```sh\n{command}\n```\n"
                "Reply `/approve`, `/approve session`, or `/deny`."
            )
        if method == "item/fileChange/requestApproval":
            return (
                f"⚠️ **Codex requests file-change approval**{reason_line}\n"
                "Reply `/approve`, `/approve session`, or `/deny`."
            )
        if method == "item/permissions/requestApproval":
            permissions = json.dumps(params.get("permissions", {}), indent=2)
            return (
                f"⚠️ **Codex requests additional permissions**{reason_line}\n"
                f"```json\n{permissions}\n```\n"
                "Reply `/approve`, `/approve session`, or `/deny`."
            )
        if method == "mcpServer/elicitation/request":
            message = str(params.get("message") or "MCP input requested")
            url = str(params.get("url") or "").strip()
            if url:
                return (
                    f"❓ **{params.get('serverName', 'MCP server')} needs input**\n"
                    f"{message}\n{url}\n"
                    "Reply `/approve` or `/deny`."
                )
            schema = json.dumps(params.get("requestedSchema", {}), indent=2)
            return (
                f"❓ **{params.get('serverName', 'MCP server')} needs form input**\n"
                f"{message}\n```json\n{schema}\n```\n"
                "Send a JSON object such as `{\"field\": \"value\"}`, or reply `/deny`."
            )
        lines = ["❓ **Codex needs input**"]
        for question in params.get("questions") or []:
            lines.append(f"\n**{question.get('header', question.get('id', 'Question'))}**")
            lines.append(str(question.get("question") or ""))
            for option in question.get("options") or []:
                lines.append(f"- {option.get('label', '')}: {option.get('description', '')}")
        lines.append("\nReply with text. For multiple questions, send `id=value | id=value`.")
        return "\n".join(lines)

    def _remove_resolved_request(self, request_id: Any) -> None:
        """Remove a request that another subscribed client already resolved."""
        for chat_id, queue in list(self.pending_server_requests.items()):
            filtered = [item for item in queue if item.request_id != request_id]
            if filtered:
                self.pending_server_requests[chat_id] = filtered
            else:
                self.pending_server_requests.pop(chat_id, None)

    def _user_message_text(self, item: dict[str, Any]) -> str:
        """Render every visible input component from a Codex user message."""
        pieces = []
        for content in item.get("content") or []:
            content_type = str(content.get("type") or "")
            if content_type == "text" and content.get("text"):
                pieces.append(str(content["text"]))
            elif content_type in {"image", "audio"} and content.get("url"):
                pieces.append(f"{content_type.title()}: {content['url']}")
            elif content_type in {"localImage", "localAudio"} and content.get("path"):
                label = "Image" if content_type == "localImage" else "Audio"
                pieces.append(f"{label}: `{content['path']}`")
            elif content_type in {"skill", "mention"} and content.get("name"):
                marker = "$" if content_type == "skill" else "@"
                pieces.append(f"{marker}{content['name']} (`{content.get('path', '')}`)")
        return "\n\n".join(pieces)

    def _task_title(self, task: dict[str, Any]) -> str:
        """Return the user-facing task title used by Codex task browsers."""
        name = str(task.get("name") or "").strip()
        if name:
            return name
        preview = str(task.get("preview") or "").strip()
        return preview.splitlines()[0][:100] if preview else "Untitled task"

    def _format_task_candidates(self, tasks: list[dict[str, Any]]) -> str:
        """Format task search results for a Discord response."""
        lines = ["**Codex tasks**"]
        for task in tasks[:20]:
            task_id = str(task.get("id") or "")
            status = str(task.get("status") or "idle")
            lines.append(f"- **{self._task_title(task)}** · `{task_id}` · {status}")
        if len(tasks) > 1:
            lines.append("\nAttach one with `/resume TASK_ID`.")
        return "\n".join(lines)

    def _help_text(self) -> str:
        """Return the Discord command contract for the transparent gateway."""
        return (
            "**Codex gateway commands**\n"
            "- `/resume` — list recent Codex tasks\n"
            "- `/resume TASK_ID-or-title` — search for and attach a task\n"
            "- `/new [/absolute/cwd]` — attach a fresh task\n"
            "- `/status` — show the attached task and live turn\n"
            "- `/refresh` — resubscribe and replay missing task history\n"
            "- `/stop` — interrupt the active Codex turn\n"
            "- `/approve`, `/approve session`, `/deny` — answer approvals\n\n"
            "A normal message starts or steers the attached Codex task. "
            "A normal message in a new Discord thread creates a Codex task."
        )

    @staticmethod
    def _escape_code_fence(text: str) -> str:
        """Prevent tool text from closing the surrounding Discord code fence."""
        return text.replace("```", "``\u200b`")


async def start_codex_daemon_gateway(adapter: Any) -> Optional[DiscordCodexGateway]:
    """Start the configured transparent bridge after Discord has connected."""
    settings = CodexGatewaySettings.load()
    if not settings.enabled:
        return None
    await _ensure_managed_codex_daemon_started(settings.socket_path)
    gateway = DiscordCodexGateway(adapter, settings)
    await gateway.start()
    return gateway


async def _ensure_managed_codex_daemon_started(socket_path: Path) -> None:
    """Start the managed daemon when the gateway uses its standard Unix socket."""
    standard_socket = (
        Path.home()
        / ".codex"
        / "app-server-control"
        / "app-server-control.sock"
    )
    if socket_path != standard_socket:
        return
    executable = shutil.which("codex")
    if executable is None:
        managed = Path.home() / ".codex" / "packages" / "standalone" / "current" / "codex"
        if managed.is_file():
            executable = str(managed)
    if executable is None:
        raise FileNotFoundError("Could not find Codex CLI for app-server daemon startup")
    process = await asyncio.create_subprocess_exec(
        executable,
        "app-server",
        "daemon",
        "start",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    if process.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not start Codex app-server daemon: {detail}")
