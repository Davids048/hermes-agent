"""Focused tests for the transparent Discord-to-Codex daemon gateway."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.codex_daemon_gateway import (
    CodexGatewaySettings,
    CodexItemDelivery,
    CodexRpcError,
    CodexTaskBinding,
    CodexTaskBindingStore,
    DiscordCodexGateway,
    PendingCodexServerRequest,
    _ensure_managed_codex_daemon_started,
)


class FakeSendResult:
    """Minimal adapter result that matches the gateway delivery contract."""

    def __init__(self, message_id: str):
        self.success = True
        self.message_id = message_id


class FakeDiscordAdapter:
    """Record Discord sends and edits without importing discord.py."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.edited: list[tuple[str, str, str, bool]] = []
        self.sent_metadata: list[dict[str, Any] | None] = []
        self.edited_metadata: list[dict[str, Any] | None] = []
        self.created_task_threads: list[tuple[str, str]] = []
        self.create_task_thread_started_event: asyncio.Event | None = None
        self.create_task_thread_wait_event: asyncio.Event | None = None
        self.renamed_threads: list[tuple[str, str, str | None]] = []
        self.registered_activities: list[tuple[str, str, str]] = []

    async def send(
        self,
        chat_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> FakeSendResult:
        """Record a new Discord message and return a stable message id."""
        self.sent.append((str(chat_id), content))
        self.sent_metadata.append(metadata)
        return FakeSendResult(str(len(self.sent)))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> FakeSendResult:
        """Record an edit against a previously returned message id."""
        self.edited.append((str(chat_id), str(message_id), content, finalize))
        self.edited_metadata.append(metadata)
        return FakeSendResult(str(message_id))

    async def create_codex_task_thread(
        self,
        parent_chat_id: str,
        name: str,
        *,
        member_user_ids: tuple[str, ...] = (),
    ) -> str:
        """Record one Discord thread created for an external Codex task."""
        if self.create_task_thread_started_event is not None:
            self.create_task_thread_started_event.set()
        if self.create_task_thread_wait_event is not None:
            await self.create_task_thread_wait_event.wait()
        self.created_task_threads.append((parent_chat_id, name))
        self.created_task_thread_members = member_user_ids
        return f"discord-task-{len(self.created_task_threads)}"

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: str | None = None,
    ) -> bool:
        """Record one Codex-driven Discord thread rename."""
        self.renamed_threads.append((thread_id, name, only_if_current_name))
        return True

    def register_codex_activity(
        self,
        message_id: str,
        collapsed_content: str,
        expanded_content: str,
    ) -> None:
        """Record one restored activity-button presentation."""
        self.registered_activities.append(
            (message_id, collapsed_content, expanded_content)
        )


class FakeCodexClient:
    """Return method-specific Codex results and record every request."""

    def __init__(self):
        """Initialize configurable app-server responses for gateway tests."""
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[Any, Any]] = []
        self.rejections: list[tuple[Any, int, str]] = []
        self.closed = False
        self.resume_turns: list[dict[str, Any]] = []
        self.history_turns: list[dict[str, Any]] = []
        self.history_pages: dict[str | None, dict[str, Any]] = {}
        self.history_list_callback = None
        self.resume_error: Exception | None = None
        self.resume_errors: list[Exception | None] = []
        self.resume_wait_event: asyncio.Event | None = None
        self.loaded_thread_ids: list[str] = []
        self.loaded_pages: dict[str | None, dict[str, Any]] = {}
        self.read_threads: dict[str, dict[str, Any]] = {}
        self.thread_start_callback = None

    async def start(self) -> None:
        """Match the production client lifecycle interface."""

    async def close(self) -> None:
        """Record gateway shutdown."""
        self.closed = True

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Return canonical fixtures for the methods exercised by the tests."""
        self.requests.append((method, params))
        if method == "thread/start":
            if self.thread_start_callback is not None:
                await self.thread_start_callback()
            return {
                "thread": {"id": "thread-created", "name": None, "preview": ""},
                "cwd": params["cwd"],
            }
        if method == "thread/name/set":
            return {}
        if method == "turn/start":
            return {"turn": {"id": "turn-created", "status": "inProgress"}}
        if method == "turn/steer":
            return {"turn": {"id": params["expectedTurnId"], "status": "inProgress"}}
        if method == "turn/interrupt":
            return {}
        if method == "thread/resume":
            if self.resume_errors:
                resume_error = self.resume_errors.pop(0)
                if resume_error is not None:
                    raise resume_error
            if self.resume_error is not None:
                raise self.resume_error
            if self.resume_wait_event is not None:
                await self.resume_wait_event.wait()
            return {
                "thread": {"id": params["threadId"], "name": "Mapped task"},
                "cwd": "/tmp/project",
                "initialTurnsPage": {"data": self.resume_turns},
            }
        if method == "thread/turns/list":
            if self.history_list_callback is not None:
                await self.history_list_callback()
            cursor = params.get("cursor")
            return self.history_pages.get(
                cursor,
                {"data": self.history_turns, "nextCursor": None},
            )
        if method == "thread/list":
            return {
                "data": [
                    {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "name": "Search result",
                        "status": "idle",
                    }
                ]
            }
        if method == "thread/loaded/list":
            return self.loaded_pages.get(
                params.get("cursor"),
                {"data": self.loaded_thread_ids, "nextCursor": None},
            )
        if method == "thread/read":
            return {"thread": self.read_threads[params["threadId"]]}
        raise AssertionError(f"Unexpected Codex method: {method}")

    async def respond(self, request_id: Any, result: Any) -> None:
        """Record one successful server-request response."""
        self.responses.append((request_id, result))

    async def reject(self, request_id: Any, code: int, message: str) -> None:
        """Record one server-request rejection."""
        self.rejections.append((request_id, code, message))


class FakeMessageEvent:
    """Construct the Discord fields consumed by the transparent gateway."""

    def __init__(
        self,
        text: str,
        *,
        chat_id: str = "discord-thread",
        message_id: str = "discord-message",
    ):
        """Build a message event for one Discord thread and author message id."""
        self.text = text
        self.message_id = message_id
        self.media_urls: list[str] = []
        self.media_types: list[str] = []
        self.source = SimpleNamespace(
            chat_id=chat_id,
            chat_name="test Discord thread",
            auto_thread_initial_name="test Discord thread",
            guild_id="guild",
            parent_chat_id="parent",
        )

    def get_command(self) -> str | None:
        """Return the leading slash command without its arguments."""
        if not self.text.startswith("/"):
            return None
        return self.text[1:].split(maxsplit=1)[0]

    def get_command_args(self) -> str:
        """Return the text after the leading slash command."""
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) == 2 else ""


class FakeDaemonStartProcess:
    """Return one successful managed-daemon startup subprocess result."""

    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return the machine-readable success response without an error."""
        return b'{"status":"alreadyRunning"}\n', b""


@pytest.fixture
def gateway(tmp_path: Path) -> DiscordCodexGateway:
    """Build a transparent gateway with isolated state and fake transports."""
    settings = CodexGatewaySettings(
        enabled=True,
        socket_path=tmp_path / "codex.sock",
        default_cwd=tmp_path,
        state_path=tmp_path / "bindings.json",
        parent_chat_id="parent-channel",
        member_user_ids=("discord-user",),
    )
    return DiscordCodexGateway(
        FakeDiscordAdapter(), settings, client=FakeCodexClient()
    )


def test_codex_gateway_settings_load_member_user_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The YAML membership list controls accounts joined to Codex task threads."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """codex_gateway:
  enabled: true
  member_user_ids:
    - "123"
    - "456"
""",
        encoding="utf-8",
    )

    settings = CodexGatewaySettings.load()

    assert settings.member_user_ids == ("123", "456")


def test_binding_store_round_trips_mapping(tmp_path: Path) -> None:
    """The Discord thread relation survives a gateway process restart."""
    path = tmp_path / "state" / "mappings.json"
    first = CodexTaskBindingStore(path)
    first.bind(
        CodexTaskBinding(
            discord_chat_id="discord-1",
            codex_thread_id="codex-1",
            cwd="/tmp/project",
            title="Training run",
            item_deliveries={
                "agent-item": CodexItemDelivery(
                    discord_message_id="discord-message",
                    final=True,
                )
            },
            pending_discord_message_ids=["pending-message"],
        )
    )

    second = CodexTaskBindingStore(path)

    assert second.bindings["discord-1"].codex_thread_id == "codex-1"
    assert second.bindings["discord-1"].title == "Training run"
    assert second.bindings["discord-1"].item_deliveries["agent-item"] == (
        CodexItemDelivery(discord_message_id="discord-message", final=True)
    )
    assert second.bindings["discord-1"].pending_discord_message_ids == [
        "pending-message"
    ]


def test_binding_store_keeps_one_discord_thread_per_codex_task(tmp_path: Path) -> None:
    """Reattaching a Codex task removes its prior Discord-thread mapping."""
    store = CodexTaskBindingStore(tmp_path / "mappings.json")
    store.bind(CodexTaskBinding("discord-1", "codex-1", "/tmp/project"))
    store.bind(CodexTaskBinding("discord-2", "codex-1", "/tmp/project"))

    assert set(store.bindings) == {"discord-2"}


@pytest.mark.asyncio
async def test_standard_socket_start_invokes_managed_codex_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway startup restores the daemon that owns the standard Unix socket."""
    launched: list[tuple[Any, ...]] = []

    async def create_process(*args: Any, **kwargs: Any) -> FakeDaemonStartProcess:
        """Record the daemon lifecycle command without starting a process."""
        launched.append((*args, kwargs))
        return FakeDaemonStartProcess()

    monkeypatch.setattr("gateway.codex_daemon_gateway.shutil.which", lambda _: "/bin/codex")
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway.asyncio.create_subprocess_exec",
        create_process,
    )
    socket_path = (
        Path.home()
        / ".codex"
        / "app-server-control"
        / "app-server-control.sock"
    )

    await _ensure_managed_codex_daemon_started(socket_path)

    assert launched[0][:4] == ("/bin/codex", "app-server", "daemon", "start")


@pytest.mark.asyncio
async def test_first_message_creates_codex_task_and_starts_turn(
    gateway: DiscordCodexGateway,
) -> None:
    """A normal message creates Codex-owned state without a Hermes agent run."""
    event = FakeMessageEvent("inspect the repository")

    response = await gateway.handle_message(event)

    assert response == ""
    assert gateway.bindings.bindings["discord-thread"].codex_thread_id == "thread-created"
    assert [method for method, _ in gateway.client.requests] == [
        "thread/start",
        "thread/name/set",
        "turn/start",
    ]
    turn_params = gateway.client.requests[-1][1]
    assert turn_params["input"] == [{"type": "text", "text": "inspect the repository"}]
    assert turn_params["clientUserMessageId"] == "discord-message"


@pytest.mark.asyncio
async def test_external_started_task_creates_discord_thread_and_mapping(
    gateway: DiscordCodexGateway,
) -> None:
    """A task started by another Codex client appears in Discord immediately."""
    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "name": None,
                "preview": "inspect the cluster",
                "cwd": "/tmp/cluster",
                "source": "vscode",
            }
        },
    )

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "inspect the cluster")
    ]
    assert gateway.adapter.created_task_thread_members == ("discord-user",)
    binding = gateway.bindings.bindings["discord-task-1"]
    assert binding.codex_thread_id == "external-thread"
    assert binding.cwd == "/tmp/cluster"
    assert [method for method, _ in gateway.client.requests[-2:]] == [
        "thread/resume",
        "thread/turns/list",
    ]


@pytest.mark.asyncio
async def test_external_started_task_waits_for_rollout_before_discord_thread(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient missing rollout delays Discord creation until resume works."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._ROLLOUT_RETRY_INITIAL_DELAY_SECONDS", 0
    )
    gateway.client.resume_errors = [
        CodexRpcError(
            -32603,
            "no rollout found for thread id external-thread",
        )
    ]
    gateway.client.resume_wait_event = asyncio.Event()

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "inspect the cluster",
                "cwd": "/tmp/cluster",
                "source": "vscode",
            }
        },
    )

    assert gateway.adapter.created_task_threads == []
    retry_task = gateway._provisional_thread_tasks["external-thread"]
    gateway.client.resume_wait_event.set()
    await asyncio.wait_for(retry_task, timeout=1)

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "inspect the cluster")
    ]
    assert gateway.bindings.bindings["discord-task-1"].codex_thread_id == (
        "external-thread"
    )
    assert [
        method for method, _ in gateway.client.requests if method == "thread/resume"
    ] == ["thread/resume", "thread/resume"]


@pytest.mark.asyncio
async def test_closed_external_task_cancels_missing_rollout_retry(
    gateway: DiscordCodexGateway,
) -> None:
    """Closing a provisional Codex task leaves no Discord thread or retry."""
    gateway.client.resume_error = CodexRpcError(
        -32603,
        "no rollout found for thread id external-thread",
    )

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )
    await gateway.handle_notification(
        "thread/closed",
        {"threadId": "external-thread"},
    )

    assert gateway.adapter.created_task_threads == []
    assert gateway.bindings.bindings == {}
    assert "external-thread" not in gateway._provisional_thread_tasks
    assert "external-thread" not in gateway._awaiting_rollout_thread_ids
    assert "external-thread" not in gateway._history_syncing_thread_ids
    assert "external-thread" not in gateway._queued_history_notifications


@pytest.mark.asyncio
async def test_closed_resumable_task_finishes_discord_thread_creation(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing after rollout durability cannot orphan Discord thread creation."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._ROLLOUT_RETRY_INITIAL_DELAY_SECONDS", 0
    )
    gateway.client.resume_errors = [
        CodexRpcError(
            -32603,
            "no rollout found for thread id external-thread",
        )
    ]
    gateway.adapter.create_task_thread_started_event = asyncio.Event()
    gateway.adapter.create_task_thread_wait_event = asyncio.Event()

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "inspect the cluster",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )
    mirror_task = gateway._provisional_thread_tasks["external-thread"]
    await asyncio.wait_for(
        gateway.adapter.create_task_thread_started_event.wait(),
        timeout=1,
    )

    await gateway.handle_notification(
        "thread/closed",
        {"threadId": "external-thread"},
    )
    assert mirror_task.cancelled() is False
    gateway.adapter.create_task_thread_wait_event.set()
    await asyncio.wait_for(mirror_task, timeout=1)

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "inspect the cluster")
    ]
    assert gateway.bindings.bindings["discord-task-1"].codex_thread_id == (
        "external-thread"
    )
    assert "external-thread" not in gateway._awaiting_rollout_thread_ids


@pytest.mark.asyncio
async def test_closed_established_task_preserves_discord_mapping(
    gateway: DiscordCodexGateway,
) -> None:
    """Closing a resumable Codex task retains its established Discord thread."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.active_turns["codex-thread"] = "active-turn"

    await gateway.handle_notification(
        "thread/closed",
        {"threadId": "codex-thread"},
    )

    assert gateway.bindings.bindings["discord-thread"].codex_thread_id == (
        "codex-thread"
    )
    assert "codex-thread" not in gateway.active_turns


@pytest.mark.asyncio
async def test_closed_deferred_task_never_starts_a_rollout_retry(
    gateway: DiscordCodexGateway,
) -> None:
    """A close event removes a task deferred behind a Discord-originated start."""
    gateway._pending_gateway_thread_starts = 1
    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )

    await gateway.handle_notification(
        "thread/closed",
        {"threadId": "external-thread"},
    )
    gateway._pending_gateway_thread_starts = 0
    await gateway._drain_deferred_started_threads()

    assert gateway.adapter.created_task_threads == []
    assert gateway.client.requests == []
    assert gateway._provisional_thread_tasks == {}
    assert gateway._awaiting_rollout_thread_ids == set()


@pytest.mark.asyncio
async def test_external_started_task_does_not_retry_other_resume_errors(
    gateway: DiscordCodexGateway,
) -> None:
    """A non-rollout RPC error ends provisional mirroring after one attempt."""
    gateway.client.resume_error = CodexRpcError(
        -32603,
        "No rollout found for thread id external-thread",
    )

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "inspect the cluster",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )

    assert gateway.adapter.created_task_threads == []
    assert "external-thread" not in gateway._provisional_thread_tasks
    assert "external-thread" not in gateway._awaiting_rollout_thread_ids
    assert [
        method for method, _ in gateway.client.requests if method == "thread/resume"
    ] == ["thread/resume"]


@pytest.mark.asyncio
async def test_external_started_task_unblocks_when_history_cleanup_fails(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup failure cannot retain a provisional task or block notifications."""
    async def fail_history_cleanup(_thread_id: str) -> None:
        """Simulate a Discord delivery failure during queued-event replay."""
        raise RuntimeError("history cleanup failed")

    monkeypatch.setattr(gateway, "_finish_history_sync", fail_history_cleanup)

    await asyncio.wait_for(
        gateway.handle_notification(
            "thread/started",
            {
                "thread": {
                    "id": "external-thread",
                    "preview": "inspect the cluster",
                    "cwd": "/tmp/cluster",
                    "source": "cli",
                }
            },
        ),
        timeout=1,
    )

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "inspect the cluster")
    ]
    assert "external-thread" not in gateway._provisional_thread_tasks
    assert "external-thread" not in gateway._awaiting_rollout_thread_ids
    assert "external-thread" not in gateway._history_syncing_thread_ids
    assert "external-thread" not in gateway._queued_history_notifications


@pytest.mark.asyncio
async def test_gateway_stop_cancels_missing_rollout_retry(
    gateway: DiscordCodexGateway,
) -> None:
    """Gateway shutdown cancels rollout polling before closing the client."""
    gateway.client.resume_error = CodexRpcError(
        -32603,
        "no rollout found for thread id external-thread",
    )
    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "preview": "",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )

    await gateway.stop()

    assert gateway.client.closed is True
    assert gateway._provisional_thread_tasks == {}
    assert gateway._awaiting_rollout_thread_ids == set()
    assert "external-thread" not in gateway._history_syncing_thread_ids


@pytest.mark.asyncio
async def test_external_started_subagent_task_creates_discord_thread(
    gateway: DiscordCodexGateway,
) -> None:
    """Every daemon-owned task appears even when its source is a subagent."""
    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "subagent-thread",
                "name": "inspect child task",
                "preview": "inspect child task",
                "cwd": "/tmp/cluster",
                "source": "subAgent",
            }
        },
    )

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "inspect child task")
    ]
    assert gateway.bindings.bindings["discord-task-1"].codex_thread_id == (
        "subagent-thread"
    )


@pytest.mark.asyncio
async def test_external_started_task_replays_full_completed_history(
    gateway: DiscordCodexGateway,
) -> None:
    """A recovered task includes every completed prompt and response."""
    gateway.client.history_pages = {
        None: {
            "data": [
                {
                    "id": "first-turn",
                    "status": "completed",
                    "items": [
                        {
                            "id": "first-prompt",
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "first question"}],
                        },
                        {
                            "id": "first-response",
                            "type": "agentMessage",
                            "text": "first answer",
                        },
                    ],
                }
            ],
            "nextCursor": "second-page",
        },
        "second-page": {
            "data": [
                {
                    "id": "completed-turn",
                    "status": "completed",
                    "items": [
                        {
                            "id": "terminal-prompt",
                            "type": "userMessage",
                            "content": [{"type": "text", "text": "what is beijing"}],
                        },
                        {
                            "id": "codex-response",
                            "type": "agentMessage",
                            "text": "Beijing is the capital city of China.",
                        },
                    ],
                }
            ],
            "nextCursor": None,
        },
    }

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "name": "what is beijing",
                "preview": "what is beijing",
                "cwd": "/tmp/cluster",
                "source": "cli",
            }
        },
    )

    assert gateway.adapter.sent == [
        ("discord-task-1", "first question"),
        ("discord-task-1", "first answer"),
        ("discord-task-1", "what is beijing"),
        ("discord-task-1", "Beijing is the capital city of China."),
    ]
    assert gateway.adapter.sent_metadata == [
        {"message_style": "codex_user_prompt"},
        None,
        {"message_style": "codex_user_prompt"},
        None,
    ]


@pytest.mark.asyncio
async def test_resume_bound_tasks_replays_missing_history_once(
    gateway: DiscordCodexGateway,
) -> None:
    """Reconnect fills a durable gap once and records its Codex item id."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.client.history_turns = [
        {
            "id": "completed-turn",
            "status": "completed",
            "items": [
                {
                    "id": "codex-response",
                    "type": "agentMessage",
                    "text": "Already mirrored response.",
                }
            ],
        }
    ]

    await gateway._resume_bound_tasks()
    await gateway._resume_bound_tasks()

    assert gateway.adapter.sent == [
        ("discord-thread", "Already mirrored response.")
    ]
    delivery = gateway.bindings.item_delivery("discord-thread", "codex-response")
    assert delivery is not None
    assert delivery.final is True


@pytest.mark.asyncio
async def test_resume_bound_tasks_replays_notification_after_history(
    gateway: DiscordCodexGateway,
) -> None:
    """A live item received during catch-up follows the durable transcript."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.client.history_turns = [
        {
            "id": "older-turn",
            "status": "completed",
            "items": [
                {
                    "id": "older-response",
                    "type": "agentMessage",
                    "text": "older response",
                }
            ],
        }
    ]
    delivered = False

    async def deliver_live_item() -> None:
        """Inject one live completion while the history request is pending."""
        nonlocal delivered
        if delivered:
            return
        delivered = True
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-thread",
                "turnId": "live-turn",
                "item": {
                    "id": "live-response",
                    "type": "agentMessage",
                    "text": "live response",
                },
            },
        )

    gateway.client.history_list_callback = deliver_live_item

    await gateway._resume_bound_tasks()

    assert gateway.adapter.sent == [
        ("discord-thread", "older response"),
        ("discord-thread", "live response"),
    ]


@pytest.mark.asyncio
async def test_reconnect_preserves_discord_prompt_identity(
    gateway: DiscordCodexGateway,
) -> None:
    """A restart cannot echo a Discord prompt that Codex accepted beforehand."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_message(FakeMessageEvent("inspect the repository"))

    restarted = DiscordCodexGateway(
        FakeDiscordAdapter(),
        gateway.settings,
        client=FakeCodexClient(),
    )
    restarted.client.history_turns = [
        {
            "id": "accepted-turn",
            "status": "completed",
            "items": [
                {
                    "id": "accepted-prompt",
                    "type": "userMessage",
                    "clientId": "discord-message",
                    "content": [
                        {"type": "text", "text": "inspect the repository"}
                    ],
                }
            ],
        }
    ]

    await restarted._resume_bound_tasks()

    assert restarted.adapter.sent == []
    delivery = restarted.bindings.item_delivery(
        "discord-thread", "accepted-prompt"
    )
    assert delivery == CodexItemDelivery(
        discord_message_id="discord-message",
        final=True,
    )
    assert restarted.bindings.bindings[
        "discord-thread"
    ].pending_discord_message_ids == []


@pytest.mark.asyncio
async def test_first_external_prompt_names_an_untitled_discord_thread(
    gateway: DiscordCodexGateway,
) -> None:
    """The first terminal prompt replaces the temporary task thread title."""
    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "name": None,
                "preview": "",
                "cwd": "/tmp/cluster",
                "source": "vscode",
            }
        },
    )
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "external-thread",
            "turnId": "turn-1",
            "item": {
                "id": "user-1",
                "type": "userMessage",
                "content": [{"type": "text", "text": "inspect the cluster\ncarefully"}],
            },
        },
    )

    assert gateway.adapter.renamed_threads == [
        ("discord-task-1", "inspect the cluster", "Untitled task")
    ]
    assert gateway.bindings.bindings["discord-task-1"].title == "inspect the cluster"


@pytest.mark.asyncio
async def test_gateway_started_task_does_not_create_second_discord_thread(
    gateway: DiscordCodexGateway,
) -> None:
    """A task created from Discord retains its original Discord thread."""
    async def announce_started_thread() -> None:
        """Deliver the local thread notification before its response."""
        await gateway.handle_notification(
            "thread/started",
            {
                "thread": {
                    "id": "thread-created",
                    "preview": "",
                    "cwd": "/tmp/project",
                    "source": "vscode",
                }
            },
        )

    gateway.client.thread_start_callback = announce_started_thread

    await gateway.handle_message(FakeMessageEvent("inspect the repository"))

    assert gateway.adapter.created_task_threads == []
    assert set(gateway.bindings.bindings) == {"discord-thread"}


@pytest.mark.asyncio
async def test_reconnect_discovers_loaded_unbound_task(
    gateway: DiscordCodexGateway,
) -> None:
    """Gateway startup recovers an external task whose start event was missed."""
    gateway.client.loaded_thread_ids = ["external-thread"]
    gateway.client.read_threads["external-thread"] = {
        "id": "external-thread",
        "name": "Recovered task",
        "preview": "",
        "cwd": "/tmp/recovered",
        "source": "cli",
    }

    await gateway._resume_bound_tasks()

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "Recovered task")
    ]
    assert gateway.bindings.bindings["discord-task-1"].codex_thread_id == (
        "external-thread"
    )


@pytest.mark.asyncio
async def test_reconnect_discovers_every_loaded_task_page(
    gateway: DiscordCodexGateway,
) -> None:
    """Daemon task discovery follows every opaque loaded-task cursor."""
    gateway.client.loaded_pages = {
        None: {"data": ["external-one"], "nextCursor": "page-two"},
        "page-two": {"data": ["external-two"], "nextCursor": None},
    }
    gateway.client.read_threads = {
        "external-one": {
            "id": "external-one",
            "name": "First task",
            "cwd": "/tmp/one",
            "source": "cli",
        },
        "external-two": {
            "id": "external-two",
            "name": "Second task",
            "cwd": "/tmp/two",
            "source": "appServer",
        },
    }

    await gateway._resume_bound_tasks()

    assert gateway.adapter.created_task_threads == [
        ("parent-channel", "First task"),
        ("parent-channel", "Second task"),
    ]


@pytest.mark.asyncio
async def test_new_command_names_created_codex_task(
    gateway: DiscordCodexGateway,
) -> None:
    """The Discord task title remains searchable after explicit task creation."""
    response = await gateway.handle_message(FakeMessageEvent("/new"))

    assert "thread-created" in response
    assert [method for method, _ in gateway.client.requests] == [
        "thread/start",
        "thread/name/set",
    ]
    assert gateway.client.requests[-1][1] == {
        "threadId": "thread-created",
        "name": "test Discord thread",
    }


@pytest.mark.asyncio
async def test_message_steers_turn_started_by_another_client(
    gateway: DiscordCodexGateway,
) -> None:
    """Discord input reaches the exact active turn learned from daemon events."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.active_turns["codex-thread"] = "external-turn"

    await gateway.handle_message(FakeMessageEvent("also check the logs"))

    assert gateway.client.requests[-1] == (
        "turn/steer",
        {
            "threadId": "codex-thread",
            "input": [{"type": "text", "text": "also check the logs"}],
            "clientUserMessageId": "discord-message",
            "expectedTurnId": "external-turn",
        },
    )


@pytest.mark.asyncio
async def test_external_user_message_is_mirrored_but_discord_echo_is_suppressed(
    gateway: DiscordCodexGateway,
) -> None:
    """Codex terminal input appears once while Discord-originated input stays once."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.submitted_client_ids.add("from-discord")
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "user-local",
                "type": "userMessage",
                "clientId": "from-discord",
                "content": [{"type": "text", "text": "local input"}],
            },
        },
    )
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-2",
            "item": {
                "id": "user-cli",
                "type": "userMessage",
                "clientId": None,
                "content": [{"type": "text", "text": "terminal input"}],
            },
        },
    )

    assert gateway.adapter.sent == [
        ("discord-thread", "terminal input")
    ]
    assert gateway.adapter.sent_metadata == [
        {"message_style": "codex_user_prompt"}
    ]


@pytest.mark.asyncio
async def test_stream_completion_replaces_preview_with_authoritative_text(
    gateway: DiscordCodexGateway,
) -> None:
    """The completed item edits one preview instead of creating duplicate output."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    await gateway.handle_notification(
        "item/agentMessage/delta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "agent-1",
            "delta": "partial",
        },
    )
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "complete response",
            },
        },
    )

    assert gateway.adapter.sent == [
        ("discord-thread", "⏳ **Codex is working…**"),
        ("discord-thread", "complete response"),
    ]
    assert gateway.adapter.edited == []


@pytest.mark.asyncio
async def test_show_item_collapses_activity_and_separates_final_answer(
    gateway: DiscordCodexGateway,
) -> None:
    """One turn keeps tool activity compact and posts its answer separately."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    for item in (
        {
            "id": "commentary-1",
            "type": "agentMessage",
            "phase": "commentary",
            "text": "I’ll inspect the repository.",
        },
        {
            "id": "command-1",
            "type": "commandExecution",
            "command": "git status --short",
            "status": "completed",
            "aggregatedOutput": " M file.txt\n",
        },
        {
            "id": "answer-1",
            "type": "agentMessage",
            "phase": "final_answer",
            "text": "The repository has one modified file.",
        },
    ):
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-thread",
                "turnId": "turn-1",
                "item": item,
            },
        )
    await gateway.handle_notification(
        "turn/completed",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "completed"},
        },
    )

    assert gateway.adapter.sent == [
        ("discord-thread", "⏳ **Codex is working…**"),
        ("discord-thread", "The repository has one modified file."),
    ]
    assert gateway.adapter.edited[-1][0:3] == (
        "discord-thread",
        "1",
        "✅ **Activity · 2 steps**\n1 command · 1 update",
    )
    expanded = gateway.adapter.edited_metadata[-1]["expanded_content"]
    assert "I’ll inspect the repository." in expanded
    assert "git status --short" in expanded
    assert " M file.txt" in expanded
    assert gateway.bindings.item_delivery(
        "discord-thread", "commentary-1"
    ).discord_message_id == "1"
    assert gateway.bindings.item_delivery(
        "discord-thread", "command-1"
    ).discord_message_id == "1"
    assert gateway.bindings.item_delivery(
        "discord-thread", "answer-1"
    ).discord_message_id == "2"


@pytest.mark.asyncio
async def test_sync_task_history_restores_aggregated_activity_controls(
    gateway: DiscordCodexGateway,
) -> None:
    """Reconnect restores Show activity data without another Discord message."""
    binding = CodexTaskBinding(
        "discord-thread",
        "codex-thread",
        "/tmp/project",
        item_deliveries={
            "commentary-1": CodexItemDelivery("activity-message", True),
            "command-1": CodexItemDelivery("activity-message", True),
            "answer-1": CodexItemDelivery("answer-message", True),
        },
    )
    gateway.bindings.bind(binding)
    gateway.client.history_turns = [
        {
            "id": "turn-1",
            "status": "completed",
            "items": [
                {
                    "id": "commentary-1",
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": "Checking repository state.",
                },
                {
                    "id": "command-1",
                    "type": "commandExecution",
                    "command": "git status --short",
                    "status": "completed",
                },
                {
                    "id": "answer-1",
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "Repository check complete.",
                },
            ],
        }
    ]

    await gateway._sync_task_history(binding)

    assert gateway.adapter.sent == []
    assert gateway.adapter.edited == []
    assert gateway.adapter.registered_activities == [
        (
            "activity-message",
            "✅ **Activity · 2 steps**\n1 command · 1 update",
            "### Codex activity · completed\n\n"
            "**Update**\nChecking repository state.\n\n---\n\n"
            "**Command · completed**\n```sh\ngit status --short\n```",
        )
    ]


@pytest.mark.asyncio
async def test_stream_delta_continues_same_message_after_restart(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconnect resumes one cumulative Discord message without lost text."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._STREAM_EDIT_INTERVAL_SECONDS", 0
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "item/agentMessage/delta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "agent-1",
            "delta": "first half",
        },
    )
    await gateway.stream_messages[("discord-thread", "agent-1")].flush_task

    restarted = DiscordCodexGateway(
        FakeDiscordAdapter(),
        gateway.settings,
        client=FakeCodexClient(),
    )
    await restarted.handle_notification(
        "item/agentMessage/delta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "agent-1",
            "delta": " and second half",
        },
    )
    await restarted.stream_messages[("discord-thread", "agent-1")].flush_task

    assert restarted.adapter.edited == [
        (
            "discord-thread",
            "1",
            "first half and second half",
            False,
        )
    ]


def test_render_activity_item_all_thread_item_types(
    gateway: DiscordCodexGateway,
) -> None:
    """Every non-message Codex transcript item has visible Discord text."""
    fixtures = [
        {"type": "commandExecution", "command": "pwd", "status": "completed"},
        {
            "type": "fileChange",
            "changes": [{"path": "/tmp/file"}],
            "status": "completed",
        },
        {
            "type": "mcpToolCall",
            "server": "notion",
            "tool": "search",
            "status": "completed",
        },
        {"type": "dynamicToolCall", "tool": "inspect", "status": "completed"},
        {"type": "hookPrompt", "fragments": [{"text": "hook text"}]},
        {
            "type": "collabAgentToolCall",
            "tool": "spawnAgent",
            "status": "completed",
            "receiverThreadIds": ["child-task"],
        },
        {
            "type": "subAgentActivity",
            "kind": "spawned",
            "agentPath": "/root/child",
            "agentThreadId": "child-task",
        },
        {"type": "webSearch", "query": "Codex app-server"},
        {"type": "imageView", "path": "/tmp/image.png"},
        {"type": "sleep", "durationMs": 1000},
        {
            "type": "imageGeneration",
            "status": "completed",
            "savedPath": "/tmp/generated.png",
        },
        {"type": "enteredReviewMode", "review": "Review changes"},
        {"type": "exitedReviewMode", "review": "Review complete"},
        {"type": "contextCompaction"},
    ]

    for item in fixtures:
        assert gateway._render_activity_item(item, final=True)


def test_user_message_text_all_input_types(
    gateway: DiscordCodexGateway,
) -> None:
    """Codex text, media, skill, and mention inputs remain visible in Discord."""
    rendered = gateway._user_message_text(
        {
            "content": [
                {"type": "text", "text": "inspect this"},
                {"type": "image", "url": "https://example.com/image.png"},
                {"type": "audio", "url": "https://example.com/audio.mp3"},
                {"type": "localImage", "path": "/tmp/image.png"},
                {"type": "localAudio", "path": "/tmp/audio.mp3"},
                {"type": "skill", "name": "review", "path": "/tmp/SKILL.md"},
                {"type": "mention", "name": "AGENTS.md", "path": "/tmp/AGENTS.md"},
            ]
        }
    )

    assert "inspect this" in rendered
    assert "https://example.com/image.png" in rendered
    assert "https://example.com/audio.mp3" in rendered
    assert "/tmp/image.png" in rendered
    assert "/tmp/audio.mp3" in rendered
    assert "$review" in rendered
    assert "@AGENTS.md" in rendered


@pytest.mark.asyncio
async def test_command_output_delta_preserves_command_and_live_output(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal output edits the command activity message without losing context."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._STREAM_EDIT_INTERVAL_SECONDS", 0
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    await gateway.handle_notification(
        "item/started",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "command-1",
                "type": "commandExecution",
                "command": "printf hello",
                "status": "inProgress",
            },
        },
    )
    await gateway.handle_notification(
        "item/commandExecution/outputDelta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "command-1",
            "delta": "hello\n",
        },
    )
    await gateway.stream_messages[("discord-thread", "command-1")].flush_task

    collapsed = gateway.adapter.edited[-1][2]
    expanded = gateway.adapter.edited_metadata[-1]["expanded_content"]
    assert collapsed == "⏳ **Activity · 1 step**\n1 command"
    assert "printf hello" in expanded
    assert "hello" in expanded


@pytest.mark.asyncio
async def test_reasoning_summary_stream_finishes_with_authoritative_parts(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-visible reasoning summaries stream and finish without raw reasoning."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._STREAM_EDIT_INTERVAL_SECONDS", 0
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    await gateway.handle_notification(
        "item/reasoning/summaryTextDelta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "reasoning-1",
            "summaryIndex": 0,
            "delta": "Checking state",
        },
    )
    await gateway.stream_messages[("discord-thread", "reasoning-1")].flush_task
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "reasoning-1",
                "type": "reasoning",
                "summary": ["Checking state", "Comparing output"],
                "content": ["private raw reasoning"],
            },
        },
    )

    assert gateway.adapter.edited[-1][0:3] == (
        "discord-thread",
        "1",
        "⏳ **Activity · 1 step**\n1 reasoning",
    )
    assert gateway.adapter.edited_metadata[-1]["expanded_content"] == (
        "### Codex activity · working\n\n"
        "**Reasoning**\nChecking state\n\nComparing output"
    )


@pytest.mark.asyncio
async def test_external_prompt_precedes_assistant_stream(
    gateway: DiscordCodexGateway,
) -> None:
    """An external prompt claims the first placeholder before a second one streams."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "external-user",
                "type": "userMessage",
                "clientId": None,
                "content": [{"type": "text", "text": "terminal input"}],
            },
        },
    )
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "assistant output",
            },
        },
    )

    assert gateway.adapter.edited == [
        (
            "discord-thread",
            "1",
            "terminal input",
            True,
        ),
    ]
    assert gateway.adapter.sent == [
        ("discord-thread", "⏳ **Codex is working…**"),
        ("discord-thread", "⏳ **Codex is working…**"),
        ("discord-thread", "assistant output"),
    ]
    assert gateway.adapter.edited_metadata[0] == {
        "message_style": "codex_user_prompt"
    }


@pytest.mark.asyncio
async def test_command_approval_uses_codex_decision_shape(
    gateway: DiscordCodexGateway,
) -> None:
    """Discord approval resolves the original app-server request id exactly once."""
    gateway.pending_server_requests["discord-thread"] = [
        PendingCodexServerRequest(
            request_id="approval-1",
            method="item/commandExecution/requestApproval",
            params={"threadId": "codex-thread", "command": "git status"},
        )
    ]

    response = await gateway.handle_message(FakeMessageEvent("/approve session"))

    assert response == "Codex request resolved with `always`."
    assert gateway.client.responses == [
        ("approval-1", {"decision": "acceptForSession"})
    ]
    assert "discord-thread" not in gateway.pending_server_requests


@pytest.mark.asyncio
async def test_server_request_resolved_removes_stale_discord_prompt(
    gateway: DiscordCodexGateway,
) -> None:
    """A request resolved by another client cannot be answered again in Discord."""
    gateway.pending_server_requests["discord-thread"] = [
        PendingCodexServerRequest(
            request_id="approval-1",
            method="item/fileChange/requestApproval",
            params={"threadId": "codex-thread"},
        )
    ]

    await gateway.handle_notification(
        "serverRequest/resolved",
        {"threadId": "codex-thread", "requestId": "approval-1"},
    )

    assert "discord-thread" not in gateway.pending_server_requests


@pytest.mark.asyncio
async def test_mcp_form_answer_returns_structured_content(
    gateway: DiscordCodexGateway,
) -> None:
    """Discord JSON input resolves an MCP form without invoking Hermes tools."""
    gateway.pending_server_requests["discord-thread"] = [
        PendingCodexServerRequest(
            request_id="mcp-1",
            method="mcpServer/elicitation/request",
            params={"threadId": "codex-thread", "mode": "form"},
        )
    ]

    response = await gateway.handle_message(FakeMessageEvent('{"project": "fv-hub"}'))

    assert response == "Codex received the MCP form response."
    assert gateway.client.responses == [
        (
            "mcp-1",
            {
                "action": "accept",
                "content": {"project": "fv-hub"},
                "_meta": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_resume_search_binds_unique_codex_task(
    gateway: DiscordCodexGateway,
) -> None:
    """A unique server-side title search attaches its Codex task id."""
    response = await gateway.handle_message(FakeMessageEvent("/resume Search result"))

    assert "11111111-1111-1111-1111-111111111111" in response
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.codex_thread_id == "11111111-1111-1111-1111-111111111111"
    assert binding.title == "Mapped task"


@pytest.mark.asyncio
async def test_resume_without_selector_lists_recent_codex_tasks(
    gateway: DiscordCodexGateway,
) -> None:
    """The registered Discord resume command lists tasks when left empty."""
    response = await gateway.handle_message(FakeMessageEvent("/resume"))

    assert "Search result" in response
    assert gateway.client.requests[-1][0] == "thread/list"


@pytest.mark.asyncio
async def test_refresh_replays_each_missing_history_item_once(
    gateway: DiscordCodexGateway,
) -> None:
    """Repeated refreshes resubscribe without duplicating delivered history."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/original",
            title="Previous task",
        )
    )
    gateway.client.history_turns = [
        {
            "id": "completed-turn",
            "status": "completed",
            "items": [
                {
                    "id": "codex-response",
                    "type": "agentMessage",
                    "text": "Recovered response.",
                }
            ],
        }
    ]

    first_response = await gateway.handle_message(FakeMessageEvent("/refresh"))
    second_response = await gateway.handle_message(FakeMessageEvent("/refresh"))

    assert first_response.startswith("Refreshed Codex task `codex-thread`.")
    assert second_response.startswith("Refreshed Codex task `codex-thread`.")
    assert "State: idle" in second_response
    assert gateway.adapter.sent == [("discord-thread", "Recovered response.")]
    assert gateway.bindings.bindings["discord-thread"].cwd == "/tmp/project"
    assert gateway.bindings.bindings["discord-thread"].title == "Mapped task"
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", "Mapped task", "Previous task")
    ]


@pytest.mark.asyncio
async def test_refresh_reports_missing_rollout_without_detaching_task(
    gateway: DiscordCodexGateway,
) -> None:
    """Refresh explains a transient rollout gap and preserves the mapping."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.client.resume_error = CodexRpcError(
        -32603,
        "no rollout found for thread id codex-thread",
    )

    response = await gateway.handle_message(FakeMessageEvent("/refresh"))

    assert response == (
        "Codex has not written this task's rollout yet. Try `/refresh` again after "
        "the first prompt starts."
    )
    assert gateway.bindings.bindings["discord-thread"].codex_thread_id == (
        "codex-thread"
    )


@pytest.mark.asyncio
async def test_refresh_requires_an_attached_codex_task(
    gateway: DiscordCodexGateway,
) -> None:
    """Refresh gives an actionable response outside a mapped Discord thread."""
    response = await gateway.handle_message(FakeMessageEvent("/refresh"))

    assert response == "This Discord thread is not attached to a Codex task."


@pytest.mark.asyncio
async def test_resume_active_task_restores_working_indicator(
    gateway: DiscordCodexGateway,
) -> None:
    """Reconnect recreates visible progress for an already running Codex turn."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway.client.resume_turns = [
        {"id": "external-turn", "status": "inProgress", "items": []}
    ]

    await gateway._resume_binding(gateway.bindings.bindings["discord-thread"])

    assert gateway.active_turns["codex-thread"] == "external-turn"
    assert gateway.adapter.sent[-1] == (
        "discord-thread",
        "⏳ **Codex is working…**",
    )


@pytest.mark.asyncio
async def test_failed_resume_restores_displaced_task_mapping(
    gateway: DiscordCodexGateway,
) -> None:
    """A failed attachment leaves the task connected to its original thread."""
    task_id = "11111111-1111-1111-1111-111111111111"
    gateway.bindings.bind(
        CodexTaskBinding("original-discord-thread", task_id, "/tmp/project")
    )
    gateway.client.resume_error = RuntimeError("resume failed")

    with pytest.raises(RuntimeError, match="resume failed"):
        await gateway.handle_message(
            FakeMessageEvent(f"/resume {task_id}", chat_id="replacement-thread")
        )

    assert set(gateway.bindings.bindings) == {"original-discord-thread"}
    assert (
        gateway.bindings.bindings["original-discord-thread"].codex_thread_id
        == task_id
    )
