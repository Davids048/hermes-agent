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
    _discord_thread_title,
    _ensure_managed_codex_daemon_started,
)
from gateway.platforms.base import utf16_len


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
        self.created_task_starters: list[tuple[str, str]] = []
        self.create_task_thread_started_event: asyncio.Event | None = None
        self.create_task_thread_wait_event: asyncio.Event | None = None
        self.renamed_threads: list[tuple[str, str, str | None]] = []
        self.rename_thread_wait_event: asyncio.Event | None = None
        self.rename_thread_result = True
        self.chat_names: dict[str, str] = {
            "discord-thread": "test Discord thread"
        }
        self.thread_name_match_results: dict[tuple[str, str], bool] = {}
        self.registered_activities: list[tuple[str, str, str]] = []
        self.codex_request_prompts: list[dict[str, Any]] = []

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
        task_title: str,
        directory: str,
        member_user_ids: tuple[str, ...] = (),
    ) -> str:
        """Record one Discord thread created for an external Codex task."""
        if self.create_task_thread_started_event is not None:
            self.create_task_thread_started_event.set()
        if self.create_task_thread_wait_event is not None:
            await self.create_task_thread_wait_event.wait()
        self.created_task_threads.append((parent_chat_id, name))
        self.created_task_starters.append((task_title, directory))
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
        if self.rename_thread_wait_event is not None:
            await self.rename_thread_wait_event.wait()
        return self.rename_thread_result

    async def get_chat_info(self, chat_id: str) -> dict[str, str]:
        """Return the raw Discord name for one mapped test thread."""
        return {
            "name": self.chat_names.get(str(chat_id), str(chat_id)),
            "type": "thread",
        }

    async def thread_name_matches(
        self,
        thread_id: str,
        expected_name: str,
    ) -> bool:
        """Return the configured live-thread name comparison result."""
        key = (str(thread_id), expected_name)
        raw_name = self.chat_names.get(str(thread_id), str(thread_id))
        return self.thread_name_match_results.get(key, raw_name == expected_name)

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

    async def send_codex_server_request(
        self,
        chat_id: str,
        prompt: str,
        actions: list[Any],
        on_action: Any,
    ) -> FakeSendResult:
        """Record a Codex component prompt and its response callback."""
        self.codex_request_prompts.append(
            {
                "chat_id": chat_id,
                "prompt": prompt,
                "actions": actions,
                "on_action": on_action,
            }
        )
        return FakeSendResult(str(len(self.codex_request_prompts)))


class FakeCodexClient:
    """Return method-specific Codex results and record every request."""

    def __init__(self):
        """Initialize configurable app-server responses for gateway tests."""
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[Any, Any]] = []
        self.respond_error: Exception | None = None
        self.rejections: list[tuple[Any, int, str]] = []
        self.closed = False
        self.resume_turns: list[dict[str, Any]] = []
        self.history_turns: list[dict[str, Any]] = []
        self.history_pages: dict[str | None, dict[str, Any]] = {}
        self.history_list_callback = None
        self.resume_error: Exception | None = None
        self.resume_errors: list[Exception | None] = []
        self.resume_results: list[dict[str, Any]] = []
        self.resume_wait_event: asyncio.Event | None = None
        self.turn_start_error: Exception | None = None
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
            if self.turn_start_error is not None:
                turn_start_error = self.turn_start_error
                self.turn_start_error = None
                raise turn_start_error
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
            if self.resume_results:
                return self.resume_results.pop(0)
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
        if self.respond_error is not None:
            raise self.respond_error
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
            discord_title="✅ Training run | tmp/project",
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
    assert second.bindings["discord-1"].discord_title == (
        "✅ Training run | tmp/project"
    )
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


def test_discord_thread_title_two_components() -> None:
    """A Discord task title includes its status, name, and directory."""
    title = _discord_thread_title(
        "Fix sync",
        "/mnt/zfs/home/junda.su/codes/new_training",
    )

    assert title == "✅ Fix sync | codes/new_training"
    assert _discord_thread_title(
        "Fix sync",
        "/mnt/zfs/home/junda.su/codes/new_training",
        working=True,
    ) == "⏳ Fix sync | codes/new_training"


def test_discord_thread_title_long_components_fit_discord_limit() -> None:
    """Title truncation preserves the name and directory components."""
    title = _discord_thread_title(
        "😀" * 80,
        "/very-long-parent-directory/very-long-working-directory",
    )

    status_and_name, directory = title.split(" | ")
    assert status_and_name.startswith("✅ ")
    assert directory.endswith("very-long-working-directory")
    assert utf16_len(title) <= 80


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
    binding = gateway.bindings.bindings["discord-thread"]
    idle_title = _discord_thread_title(
        "test Discord thread",
        str(gateway.settings.default_cwd),
    )
    working_title = _discord_thread_title(
        "test Discord thread",
        str(gateway.settings.default_cwd),
        working=True,
    )
    assert binding.codex_thread_id == "thread-created"
    assert binding.title == "test Discord thread"
    assert binding.discord_title == working_title
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", idle_title, "test Discord thread"),
        ("discord-thread", working_title, idle_title),
    ]
    assert [method for method, _ in gateway.client.requests] == [
        "thread/start",
        "thread/name/set",
        "turn/start",
    ]
    turn_params = gateway.client.requests[-1][1]
    assert turn_params["input"] == [{"type": "text", "text": "inspect the repository"}]
    assert turn_params["clientUserMessageId"] == "discord-message"


@pytest.mark.asyncio
async def test_send_event_to_codex_thread_rename_timeout_does_not_block_submission(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited title update cannot hold Discord message cleanup open."""
    idle_title = "✅ Task name | tmp/project"
    working_title = "⏳ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=idle_title,
        )
    )
    gateway.adapter.rename_thread_wait_event = asyncio.Event()
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._DISCORD_THREAD_RENAME_TIMEOUT_SECONDS",
        0.01,
    )

    response = await asyncio.wait_for(
        gateway.handle_message(FakeMessageEvent("continue the task")),
        timeout=0.2,
    )

    assert response == ""
    assert gateway.client.requests[-1][0] == "turn/start"
    assert gateway.active_turns["codex-thread"] == "turn-created"
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title)
    ]
    assert gateway.bindings.bindings["discord-thread"].discord_title == idle_title


@pytest.mark.asyncio
async def test_create_task_for_event_uses_raw_discord_name_guard(
    gateway: DiscordCodexGateway,
) -> None:
    """Initial binding compares renames against the raw Discord thread name."""
    event = FakeMessageEvent("inspect the repository")
    event.source.chat_name = "Test server / #m7 / raw thread name"
    event.source.auto_thread_initial_name = None
    gateway.adapter.chat_names["discord-thread"] = "raw thread name"

    await gateway.handle_message(event)

    idle_title = _discord_thread_title(
        "Test server / #m7 / raw thread name",
        str(gateway.settings.default_cwd),
    )
    working_title = _discord_thread_title(
        "Test server / #m7 / raw thread name",
        str(gateway.settings.default_cwd),
        working=True,
    )
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", idle_title, "raw thread name"),
        ("discord-thread", working_title, idle_title),
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Test server / #m7 / raw thread name"
    assert binding.discord_title == working_title


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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
    ]
    assert gateway.adapter.created_task_starters == [
        ("Mapped task", "tmp/project")
    ]
    assert gateway.adapter.created_task_thread_members == ("discord-user",)
    binding = gateway.bindings.bindings["discord-task-1"]
    assert binding.codex_thread_id == "external-thread"
    assert binding.cwd == "/tmp/project"
    assert binding.title == "Mapped task"
    assert binding.discord_title == "✅ Mapped task | tmp/project"
    assert [method for method, _ in gateway.client.requests[-2:]] == [
        "thread/resume",
        "thread/turns/list",
    ]


@pytest.mark.asyncio
async def test_external_started_task_uses_resumed_name_and_directory(
    gateway: DiscordCodexGateway,
) -> None:
    """Discord creation uses metadata read after the rollout becomes durable."""
    gateway.client.resume_results = [
        {
            "thread": {
                "id": "external-thread",
                "name": "Current task name",
                "cwd": "/current/project",
            },
            "cwd": "/current/project",
            "initialTurnsPage": {
                "data": [
                    {"id": "active-turn", "status": "inProgress", "items": []}
                ]
            },
        }
    ]

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                "name": None,
                "preview": "",
                "cwd": "/stale/project",
                "source": "vscode",
            }
        },
    )

    assert gateway.adapter.created_task_threads == [
        (
            "parent-channel",
            "⏳ Current task name | current/project",
        )
    ]
    assert gateway.adapter.created_task_starters == [
        ("Current task name", "current/project")
    ]
    binding = gateway.bindings.bindings["discord-task-1"]
    assert binding.title == "Current task name"
    assert binding.cwd == "/current/project"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("started_metadata", "resumed_metadata", "expected_name"),
    [
        (
            {"name": None, "preview": "Start preview"},
            {"name": None},
            "Start preview",
        ),
        (
            {"name": "Start name", "preview": "Start preview"},
            {"preview": "Resumed preview"},
            "Start name",
        ),
    ],
)
async def test_external_started_task_merges_partial_resume_metadata(
    gateway: DiscordCodexGateway,
    started_metadata: dict[str, Any],
    resumed_metadata: dict[str, Any],
    expected_name: str,
) -> None:
    """A partial resume response preserves each omitted start-event field."""
    gateway.client.resume_results = [
        {
            "thread": {"id": "external-thread", **resumed_metadata},
            "cwd": "/current/project",
            "initialTurnsPage": {"data": []},
        }
    ]

    await gateway.handle_notification(
        "thread/started",
        {
            "thread": {
                "id": "external-thread",
                **started_metadata,
                "cwd": "/stale/project",
                "source": "vscode",
            }
        },
    )

    assert gateway.adapter.created_task_threads == [
        (
            "parent-channel",
            f"✅ {expected_name} | current/project",
        )
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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
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
async def test_resume_bound_tasks_keeps_other_task_notifications_live(
    gateway: DiscordCodexGateway,
) -> None:
    """Synchronizing one task does not defer another mapped task's output."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-first", "codex-first", "/tmp/first")
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-second", "codex-second", "/tmp/second")
    )
    delivered = False

    async def deliver_other_task_item() -> None:
        """Inject output for the task that is not synchronizing history."""
        nonlocal delivered
        if delivered:
            return
        delivered = True
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-second",
                "turnId": "live-turn",
                "item": {
                    "id": "live-response",
                    "type": "agentMessage",
                    "text": "live response",
                },
            },
        )
        assert gateway.adapter.sent == [("discord-second", "live response")]

    gateway.client.history_list_callback = deliver_other_task_item

    await gateway._resume_bound_tasks()

    assert gateway.adapter.sent == [("discord-second", "live response")]


@pytest.mark.asyncio
async def test_resume_bound_tasks_subscribes_every_task_before_history_replay(
    gateway: DiscordCodexGateway,
) -> None:
    """Startup subscribes and refreshes every title before replaying history."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-first",
            "codex-first",
            "/tmp/first",
            title="First task",
            discord_title="⏳ First task | tmp/first",
        )
    )
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-second",
            "codex-second",
            "/tmp/second",
            title="Second task",
            discord_title="⏳ Second task | tmp/second",
        )
    )
    checked_first_history_request = False

    async def verify_subscriptions_and_titles() -> None:
        """Check startup phase ordering when the first history request begins."""
        nonlocal checked_first_history_request
        if checked_first_history_request:
            return
        checked_first_history_request = True
        resumed_thread_ids = [
            params["threadId"]
            for method, params in gateway.client.requests
            if method == "thread/resume"
        ]
        assert resumed_thread_ids == ["codex-first", "codex-second"]
        assert gateway.bindings.bindings["discord-second"].discord_title == (
            "✅ Mapped task | tmp/project"
        )

    gateway.client.history_list_callback = verify_subscriptions_and_titles

    await gateway._resume_bound_tasks()

    assert checked_first_history_request is True


@pytest.mark.asyncio
async def test_finish_history_sync_drains_new_completions_in_order(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion received during queue replay follows earlier output."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway._history_syncing_thread_ids.add("codex-thread")
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "first response",
            },
        },
    )
    first_send_started = asyncio.Event()
    release_first_send = asyncio.Event()
    original_send = gateway.adapter.send

    async def block_first_send(
        chat_id: str,
        content: str,
        **kwargs: Any,
    ) -> FakeSendResult:
        """Pause the first queued delivery while a newer event arrives."""
        if content == "first response":
            first_send_started.set()
            await release_first_send.wait()
        return await original_send(chat_id, content, **kwargs)

    monkeypatch.setattr(gateway.adapter, "send", block_first_send)
    finish_task = asyncio.create_task(gateway._finish_history_sync("codex-thread"))
    await asyncio.wait_for(first_send_started.wait(), timeout=1)

    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-2",
            "item": {
                "id": "agent-2",
                "type": "agentMessage",
                "text": "second response",
            },
        },
    )
    release_first_send.set()
    await asyncio.wait_for(finish_task, timeout=1)

    assert gateway.adapter.sent == [
        ("discord-thread", "first response"),
        ("discord-thread", "second response"),
    ]
    assert "codex-thread" not in gateway._history_syncing_thread_ids
    assert "codex-thread" not in gateway._queued_history_notifications


@pytest.mark.asyncio
async def test_finish_history_sync_preserves_failed_completion_for_later_sync(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed queued delivery remains retryable by later synchronization."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway._history_syncing_thread_ids.add("codex-thread")
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "retryable response",
            },
        },
    )
    original_send = gateway.adapter.send

    async def fail_send(*args: Any, **kwargs: Any) -> FakeSendResult:
        """Fail the first attempt to deliver the queued completion."""
        raise RuntimeError("Discord delivery failed")

    monkeypatch.setattr(gateway.adapter, "send", fail_send)

    with pytest.raises(RuntimeError, match="Discord delivery failed"):
        await gateway._finish_history_sync("codex-thread")

    assert "codex-thread" not in gateway._history_syncing_thread_ids
    queued = gateway._queued_history_notifications["codex-thread"]
    assert [notification.method for notification in queued] == ["item/completed"]

    monkeypatch.setattr(gateway.adapter, "send", original_send)
    gateway._history_syncing_thread_ids.add("codex-thread")
    await gateway._finish_history_sync("codex-thread")

    assert gateway.adapter.sent == [("discord-thread", "retryable response")]
    assert "codex-thread" not in gateway._queued_history_notifications


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
    gateway.client.resume_results = [
        {
            "thread": {
                "id": "external-thread",
                "name": None,
                "preview": "",
            },
            "cwd": "/tmp/cluster",
            "initialTurnsPage": {"data": []},
        }
    ]
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
        (
            "discord-task-1",
            "✅ inspect the cluster | tmp/cluster",
            "✅ Untitled task | tmp/cluster",
        )
    ]
    assert gateway.bindings.bindings["discord-task-1"].title == "inspect the cluster"
    assert gateway.bindings.bindings["discord-task-1"].discord_title == (
        "✅ inspect the cluster | tmp/cluster"
    )


@pytest.mark.asyncio
async def test_handle_notification_thread_name_updated_omits_session_id(
    gateway: DiscordCodexGateway,
) -> None:
    """A Codex semantic name update renders only the name and directory."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Original name",
            discord_title="✅ Original name | tmp/project | codex-thread",
        )
    )

    await gateway.handle_notification(
        "thread/name/updated",
        {"threadId": "codex-thread", "threadName": "Renamed task"},
    )

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Renamed task | tmp/project",
            "✅ Original name | tmp/project | codex-thread",
        )
    ]
    assert gateway.bindings.bindings["discord-thread"].title == "Renamed task"
    assert gateway.bindings.bindings["discord-thread"].discord_title == (
        "✅ Renamed task | tmp/project"
    )


@pytest.mark.asyncio
async def test_turn_lifecycle_updates_discord_title_status(
    gateway: DiscordCodexGateway,
) -> None:
    """An active turn uses an hourglass and a completed turn uses a check mark."""
    idle_title = "✅ Task name | tmp/project"
    working_title = "⏳ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=idle_title,
        )
    )

    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )
    await gateway.handle_notification(
        "turn/completed",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "completed"},
        },
    )

    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title),
        ("discord-thread", idle_title, working_title),
    ]
    assert gateway.bindings.bindings["discord-thread"].discord_title == idle_title


@pytest.mark.asyncio
async def test_turn_lifecycle_updates_title_during_history_sync(
    gateway: DiscordCodexGateway,
) -> None:
    """Live lifecycle events update the task title during transcript replay."""
    idle_title = "✅ Task name | tmp/project"
    working_title = "⏳ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=idle_title,
        )
    )
    gateway._history_syncing_thread_ids.add("codex-thread")

    await gateway.handle_notification(
        "turn/started",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "inProgress"},
        },
    )

    assert gateway.active_turns == {"codex-thread": "turn-1"}
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title)
    ]

    await gateway.handle_notification(
        "turn/completed",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "completed"},
        },
    )

    assert gateway.active_turns == {}
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title),
        ("discord-thread", idle_title, working_title),
    ]
    queued = gateway._queued_history_notifications["codex-thread"]
    assert len(queued) == 1
    assert queued[0].method == "turn/completed"
    assert queued[0].finalize_turn_activity is True

    await gateway._finish_history_sync("codex-thread")

    assert len(gateway.adapter.renamed_threads) == 2
    assert "codex-thread" not in gateway._queued_history_notifications


@pytest.mark.asyncio
async def test_agent_message_streams_during_history_sync(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mapped task streams live text while durable history is replayed."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._STREAM_EDIT_INTERVAL_SECONDS", 0
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway._history_syncing_thread_ids.add("codex-thread")

    await gateway.handle_notification(
        "item/agentMessage/delta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "agent-1",
            "delta": "live text",
        },
    )
    await gateway.stream_messages[("discord-thread", "agent-1")].flush_task

    assert gateway.adapter.sent == [("discord-thread", "live text")]
    assert gateway._queued_history_notifications == {}

    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "complete text",
            },
        },
    )

    assert gateway.adapter.edited == []
    assert gateway._queued_history_notifications["codex-thread"][0].method == (
        "item/completed"
    )

    await gateway._finish_history_sync("codex-thread")

    assert gateway.adapter.edited == [
        ("discord-thread", "1", "complete text", True)
    ]


@pytest.mark.asyncio
async def test_user_prompt_precedes_agent_stream_during_history_sync(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal prompt reaches Discord before its same-turn agent stream."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._STREAM_EDIT_INTERVAL_SECONDS", 0
    )
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )
    gateway._history_syncing_thread_ids.add("codex-thread")

    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "user-1",
                "type": "userMessage",
                "content": [{"type": "text", "text": "terminal prompt"}],
            },
        },
    )
    await gateway.handle_notification(
        "item/agentMessage/delta",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "itemId": "agent-1",
            "delta": "partial answer",
        },
    )
    await gateway.stream_messages[("discord-thread", "agent-1")].flush_task
    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "agent-1",
                "type": "agentMessage",
                "text": "complete answer",
            },
        },
    )

    assert gateway.adapter.sent == [
        ("discord-thread", "terminal prompt"),
        ("discord-thread", "partial answer"),
    ]
    assert gateway.adapter.edited == []

    await gateway._finish_history_sync("codex-thread")

    assert gateway.adapter.sent == [
        ("discord-thread", "terminal prompt"),
        ("discord-thread", "partial answer"),
    ]
    assert gateway.adapter.edited == [
        ("discord-thread", "2", "complete answer", True)
    ]


@pytest.mark.asyncio
async def test_name_updates_preserve_guard_after_manual_discord_rename(
    gateway: DiscordCodexGateway,
) -> None:
    """A manual Discord title keeps blocking later Codex-driven renames."""
    original_discord_title = "✅ Original name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Original name",
            discord_title=original_discord_title,
        )
    )
    gateway.adapter.rename_thread_result = False

    await gateway.handle_notification(
        "thread/name/updated",
        {"threadId": "codex-thread", "threadName": "First Codex name"},
    )
    await gateway.handle_notification(
        "thread/name/updated",
        {"threadId": "codex-thread", "threadName": "Second Codex name"},
    )

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ First Codex name | tmp/project",
            original_discord_title,
        ),
        (
            "discord-thread",
            "✅ Second Codex name | tmp/project",
            original_discord_title,
        ),
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Second Codex name"
    assert binding.discord_title == original_discord_title


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
        (
            "parent-channel",
            "✅ Mapped task | tmp/project",
        )
    ]
    assert gateway.bindings.bindings["discord-task-1"].codex_thread_id == (
        "external-thread"
    )


@pytest.mark.asyncio
async def test_resume_bound_tasks_migrates_legacy_discord_title(
    gateway: DiscordCodexGateway,
) -> None:
    """Reconnect upgrades a stored name-only Discord title without clobbering it."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/original",
            title="Previous task",
        )
    )

    await gateway._resume_bound_tasks()

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Mapped task | tmp/project",
            "Previous task",
        )
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Mapped task"
    assert binding.cwd == "/tmp/project"
    assert binding.discord_title == "✅ Mapped task | tmp/project"


@pytest.mark.asyncio
async def test_resume_bound_tasks_preserves_manually_changed_legacy_title(
    gateway: DiscordCodexGateway,
) -> None:
    """Reconnect retains the legacy rename guard when Discord rejects a rename."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/original",
            title="Previous task",
        )
    )
    gateway.adapter.rename_thread_result = False

    await gateway._resume_bound_tasks()
    await gateway.handle_notification(
        "thread/name/updated",
        {"threadId": "codex-thread", "threadName": "Later task name"},
    )

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Mapped task | tmp/project",
            "Previous task",
        ),
        (
            "discord-thread",
            "✅ Later task name | tmp/project",
            "Previous task",
        ),
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Later task name"
    assert binding.discord_title == "Previous task"


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
        ("parent-channel", "✅ Mapped task | tmp/project"),
        ("parent-channel", "✅ Mapped task | tmp/project"),
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
    expected_title = _discord_thread_title(
        "test Discord thread",
        str(gateway.settings.default_cwd),
    )
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", expected_title, "test Discord thread")
    ]


@pytest.mark.asyncio
async def test_new_from_command_uses_raw_discord_name_guard(
    gateway: DiscordCodexGateway,
) -> None:
    """A first `/new` mapping separates semantic and raw Discord names."""
    event = FakeMessageEvent("/new")
    event.source.chat_name = "Test server / #m7 / raw thread name"
    gateway.adapter.chat_names["discord-thread"] = "raw thread name"

    await gateway.handle_message(event)

    assert gateway.client.requests[-1] == (
        "thread/name/set",
        {
            "threadId": "thread-created",
            "name": "Test server / #m7 / raw thread name",
        },
    )
    expected_title = _discord_thread_title(
        "Test server / #m7 / raw thread name",
        str(gateway.settings.default_cwd),
    )
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", expected_title, "raw thread name")
    ]


@pytest.mark.asyncio
async def test_new_command_reuses_semantic_name_from_rendered_discord_title(
    gateway: DiscordCodexGateway,
) -> None:
    """An unchanged contextual guard preserves the prior semantic task name."""
    previous_discord_title = "Test server / #m7 / raw thread name"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "previous-thread",
            "/tmp/previous",
            title="Semantic task",
            discord_title=previous_discord_title,
        )
    )
    event = FakeMessageEvent("/new")
    event.source.chat_name = "Test server / #raw thread name"
    gateway.adapter.chat_names["discord-thread"] = "raw thread name"
    gateway.adapter.thread_name_match_results[
        ("discord-thread", previous_discord_title)
    ] = True

    await gateway.handle_message(event)

    assert gateway.client.requests[-1] == (
        "thread/name/set",
        {"threadId": "thread-created", "name": "Semantic task"},
    )
    expected_title = _discord_thread_title(
        "Semantic task",
        str(gateway.settings.default_cwd),
    )
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", expected_title, previous_discord_title)
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Semantic task"
    assert binding.discord_title == expected_title


@pytest.mark.asyncio
async def test_new_command_uses_manual_discord_title_as_semantic_name(
    gateway: DiscordCodexGateway,
) -> None:
    """Starting another task carries a manual Discord name into Codex."""
    previous_discord_title = "Test server / #m7 / raw thread name"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "previous-thread",
            "/tmp/previous",
            title="Semantic task",
            discord_title=previous_discord_title,
        )
    )
    gateway.adapter.rename_thread_result = False
    event = FakeMessageEvent("/new")
    event.source.chat_name = "Test server / #Human title"
    gateway.adapter.chat_names["discord-thread"] = "Human title"
    gateway.adapter.thread_name_match_results[
        ("discord-thread", previous_discord_title)
    ] = False

    await gateway.handle_message(event)

    assert gateway.client.requests[-1] == (
        "thread/name/set",
        {"threadId": "thread-created", "name": "Human title"},
    )
    expected_title = _discord_thread_title(
        "Human title",
        str(gateway.settings.default_cwd),
    )
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", expected_title, previous_discord_title)
    ]
    binding = gateway.bindings.bindings["discord-thread"]
    assert binding.title == "Human title"
    assert binding.discord_title == previous_discord_title


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
async def test_message_recovers_active_turn_after_stale_start_failure(
    gateway: DiscordCodexGateway,
) -> None:
    """A failed start resumes and steers an active turn without lock re-entry."""
    idle_title = "✅ Task name | tmp/project"
    working_title = "⏳ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=idle_title,
        )
    )
    gateway.client.turn_start_error = CodexRpcError(-32603, "stale turn state")
    gateway.client.resume_turns = [
        {"id": "active-turn", "status": "inProgress", "items": []}
    ]

    response = await asyncio.wait_for(
        gateway.handle_message(FakeMessageEvent("continue the task")),
        timeout=1,
    )

    assert response == ""
    assert [method for method, _ in gateway.client.requests[-3:]] == [
        "turn/start",
        "thread/resume",
        "turn/steer",
    ]
    assert gateway.client.requests[-1][1]["expectedTurnId"] == "active-turn"
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title)
    ]


@pytest.mark.asyncio
async def test_interrupt_updates_title_for_discovered_active_turn(
    gateway: DiscordCodexGateway,
) -> None:
    """Interrupt marks the title working when resume discovers an active turn."""
    idle_title = "✅ Task name | tmp/project"
    working_title = "⏳ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=idle_title,
        )
    )
    gateway.client.resume_turns = [
        {"id": "active-turn", "status": "inProgress", "items": []}
    ]

    response = await gateway.handle_message(FakeMessageEvent("/interrupt"))

    assert response == "Interrupt requested for Codex turn `active-turn`."
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", working_title, idle_title)
    ]
    assert gateway.client.requests[-1] == (
        "turn/interrupt",
        {"threadId": "codex-thread", "turnId": "active-turn"},
    )


@pytest.mark.asyncio
async def test_interrupt_updates_title_for_discovered_idle_task(
    gateway: DiscordCodexGateway,
) -> None:
    """Interrupt restores the check mark when resume finds no active turn."""
    working_title = "⏳ Task name | tmp/project"
    idle_title = "✅ Task name | tmp/project"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/project",
            title="Task name",
            discord_title=working_title,
        )
    )

    response = await gateway.handle_message(FakeMessageEvent("/interrupt"))

    assert response == "The Codex task is idle."
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", idle_title, working_title)
    ]


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
async def test_set_turn_activity_detail_locked_coalesces_completed_items(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A burst of completed activity creates one state save and Discord edit."""
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
    save_count = 0
    original_save = gateway.bindings._save

    def count_save() -> None:
        """Count persistent mapping writes while preserving their behavior."""
        nonlocal save_count
        save_count += 1
        original_save()

    monkeypatch.setattr(gateway.bindings, "_save", count_save)
    for index in range(100):
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-thread",
                "turnId": "turn-1",
                "item": {
                    "id": f"commentary-{index}",
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": f"Update {index}",
                },
            },
        )

    assert gateway.adapter.edited == []
    assert save_count == 0

    await gateway.handle_notification(
        "item/completed",
        {
            "threadId": "codex-thread",
            "turnId": "turn-1",
            "item": {
                "id": "answer-1",
                "type": "agentMessage",
                "phase": "final_answer",
                "text": "All updates completed.",
            },
        },
    )

    assert len(gateway.adapter.edited) == 1
    assert gateway.adapter.edited[0][2] == (
        "⏳ **Activity · 100 steps**\n100 updates"
    )
    assert gateway.adapter.sent[-1] == (
        "discord-thread",
        "All updates completed.",
    )
    assert save_count == 2
    for index in range(100):
        assert gateway.bindings.item_delivery(
            "discord-thread", f"commentary-{index}"
        ) == CodexItemDelivery(discord_message_id="1", final=True)

    await gateway.handle_notification(
        "turn/completed",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "completed"},
        },
    )


@pytest.mark.asyncio
async def test_flush_turn_activity_publishes_periodic_batches(
    gateway: DiscordCodexGateway,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-running turn publishes one edit for each elapsed activity interval."""
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._ACTIVITY_EDIT_INTERVAL_SECONDS", 0
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

    for index in range(3):
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-thread",
                "turnId": "turn-1",
                "item": {
                    "id": f"first-update-{index}",
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": f"First update {index}",
                },
            },
        )
    display = gateway.turn_displays[("discord-thread", "turn-1")]
    if display.flush_task is not None:
        await display.flush_task

    assert len(gateway.adapter.edited) == 1
    assert gateway.adapter.edited[-1][2] == (
        "⏳ **Activity · 3 steps**\n3 updates"
    )

    for index in range(2):
        await gateway.handle_notification(
            "item/completed",
            {
                "threadId": "codex-thread",
                "turnId": "turn-1",
                "item": {
                    "id": f"second-update-{index}",
                    "type": "agentMessage",
                    "phase": "commentary",
                    "text": f"Second update {index}",
                },
            },
        )
    display = gateway.turn_displays[("discord-thread", "turn-1")]
    if display.flush_task is not None:
        await display.flush_task

    assert len(gateway.adapter.edited) == 2
    assert gateway.adapter.edited[-1][2] == (
        "⏳ **Activity · 5 steps**\n5 updates"
    )

    await gateway.handle_notification(
        "turn/completed",
        {
            "threadId": "codex-thread",
            "turn": {"id": "turn-1", "status": "completed"},
        },
    )


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
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._ACTIVITY_EDIT_INTERVAL_SECONDS", 0
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
    display = gateway.turn_displays[("discord-thread", "turn-1")]
    if display.flush_task is not None:
        await display.flush_task

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
    monkeypatch.setattr(
        "gateway.codex_daemon_gateway._ACTIVITY_EDIT_INTERVAL_SECONDS", 0
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
    display = gateway.turn_displays[("discord-thread", "turn-1")]
    if display.flush_task is not None:
        await display.flush_task
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
    display = gateway.turn_displays[("discord-thread", "turn-1")]
    if display.flush_task is not None:
        await display.flush_task

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
async def test_handle_server_request_empty_mcp_form_maps_allow_to_empty_content(
    gateway: DiscordCodexGateway,
) -> None:
    """An empty MCP form renders controls and returns an empty object on Allow."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "mcp-1",
        "mcpServer/elicitation/request",
        {
            "threadId": "codex-thread",
            "mode": "form",
            "serverName": "node_repl",
            "message": 'Allow Computer Use to use "TickTick"?',
            "requestedSchema": {"type": "object", "properties": {}},
        },
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    assert [action.label for action in prompt["actions"]] == ["Allow", "Deny"]
    assert await prompt["on_action"](prompt["actions"][0]) is True
    assert gateway.client.responses == [
        ("mcp-1", {"action": "accept", "content": {}, "_meta": None})
    ]
    assert "discord-thread" not in gateway.pending_server_requests


@pytest.mark.asyncio
async def test_handle_server_request_command_approval_maps_session_decision(
    gateway: DiscordCodexGateway,
) -> None:
    """A Discord approval button returns the matching Codex decision value."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "approval-1",
        "item/commandExecution/requestApproval",
        {"threadId": "codex-thread", "command": "git status"},
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    session_action = next(
        action for action in prompt["actions"] if action.label == "Approve Session"
    )
    assert await prompt["on_action"](session_action) is True
    assert gateway.client.responses == [
        ("approval-1", {"decision": "acceptForSession"})
    ]


@pytest.mark.asyncio
async def test_handle_server_request_uses_only_available_command_decisions(
    gateway: DiscordCodexGateway,
) -> None:
    """Discord presents and accepts only decisions offered by Codex."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "approval-1",
        "item/commandExecution/requestApproval",
        {
            "threadId": "codex-thread",
            "command": "git status",
            "availableDecisions": ["accept", "decline"],
        },
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    assert [action.label for action in prompt["actions"]] == [
        "Approve Once",
        "Deny",
    ]
    response = await gateway.handle_message(FakeMessageEvent("/approve session"))
    assert response == "Codex offered these responses: `/approve`, `/deny`"
    assert gateway.client.responses == []


@pytest.mark.asyncio
async def test_handle_server_request_mcp_enum_maps_selected_content(
    gateway: DiscordCodexGateway,
) -> None:
    """An MCP enum choice returns an object keyed by the schema field name."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "mcp-enum-1",
        "mcpServer/elicitation/request",
        {
            "threadId": "codex-thread",
            "mode": "form",
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "permission": {"type": "string", "enum": ["Allow", "Deny"]}
                },
            },
        },
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    assert [action.label for action in prompt["actions"]] == [
        "Allow",
        "Deny",
        "Decline request",
    ]
    assert await prompt["on_action"](prompt["actions"][0]) is True
    assert gateway.client.responses == [
        (
            "mcp-enum-1",
            {
                "action": "accept",
                "content": {"permission": "Allow"},
                "_meta": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_handle_server_request_mcp_titled_choice_maps_const_value(
    gateway: DiscordCodexGateway,
) -> None:
    """An MCP titled choice displays its title and returns its const value."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "mcp-titled-1",
        "mcpServer/elicitation/request",
        {
            "threadId": "codex-thread",
            "mode": "form",
            "requestedSchema": {
                "type": "object",
                "properties": {
                    "permission": {
                        "type": "string",
                        "oneOf": [
                            {"const": "allow_once", "title": "Allow Once"},
                            {"const": "deny", "title": "Deny"},
                        ],
                    }
                },
            },
        },
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    assert [action.label for action in prompt["actions"]] == [
        "Allow Once",
        "Deny",
        "Decline request",
    ]
    assert await prompt["on_action"](prompt["actions"][0]) is True
    assert gateway.client.responses == [
        (
            "mcp-titled-1",
            {
                "action": "accept",
                "content": {"permission": "allow_once"},
                "_meta": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_resolve_pending_request_omits_null_permission_fields(
    gateway: DiscordCodexGateway,
) -> None:
    """A permission grant excludes null fields from the Codex response."""
    gateway.pending_server_requests["discord-thread"] = [
        PendingCodexServerRequest(
            request_id="permission-1",
            method="item/permissions/requestApproval",
            params={
                "permissions": {
                    "network": None,
                    "fileSystem": {"read": ["/tmp/project"]},
                }
            },
        )
    ]

    response = await gateway.handle_message(FakeMessageEvent("/approve"))

    assert response == "Codex request resolved with `approve`."
    assert gateway.client.responses == [
        (
            "permission-1",
            {
                "permissions": {"fileSystem": {"read": ["/tmp/project"]}},
                "scope": "turn",
            },
        )
    ]


@pytest.mark.asyncio
async def test_handle_server_request_single_question_maps_choice_answer(
    gateway: DiscordCodexGateway,
) -> None:
    """A single Codex question maps each declared option to a Discord button."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    await gateway.handle_server_request(
        "question-1",
        "item/tool/requestUserInput",
        {
            "threadId": "codex-thread",
            "questions": [
                {
                    "id": "format",
                    "header": "Format",
                    "question": "Choose the response format.",
                    "options": [
                        {"label": "Compact", "description": "One paragraph."},
                        {"label": "Detailed", "description": "Several sections."},
                    ],
                }
            ],
        },
    )

    prompt = gateway.adapter.codex_request_prompts[0]
    assert [action.label for action in prompt["actions"]] == [
        "Compact",
        "Detailed",
    ]
    assert await prompt["on_action"](prompt["actions"][1]) is True
    assert gateway.client.responses == [
        (
            "question-1",
            {"answers": {"format": {"answers": ["Detailed"]}}},
        )
    ]


@pytest.mark.asyncio
async def test_resolve_server_request_action_stale_request_preserves_queue(
    gateway: DiscordCodexGateway,
) -> None:
    """A disconnected socket's button cannot resolve a same-id replacement."""
    stale_request = PendingCodexServerRequest(
        request_id="approval-1",
        method="item/fileChange/requestApproval",
        params={"threadId": "codex-thread"},
    )
    gateway.pending_server_requests["discord-thread"] = [stale_request]
    await gateway._clear_socket_scoped_state()
    replacement_request = PendingCodexServerRequest(
        request_id="approval-1",
        method="item/fileChange/requestApproval",
        params={"threadId": "codex-thread"},
    )
    gateway.pending_server_requests["discord-thread"] = [replacement_request]
    stale_action = gateway._server_request_actions(
        "item/fileChange/requestApproval", {}
    )[0]

    resolved = await gateway._resolve_server_request_action(
        "discord-thread", stale_request, stale_action
    )

    assert resolved is False
    assert gateway.client.responses == []
    assert gateway.pending_server_requests["discord-thread"][0] is replacement_request


@pytest.mark.asyncio
async def test_resolve_server_request_action_failure_keeps_request_for_retry(
    gateway: DiscordCodexGateway,
) -> None:
    """A failed Codex response write leaves the same request pending."""
    pending_request = PendingCodexServerRequest(
        request_id="mcp-1",
        method="mcpServer/elicitation/request",
        params={"mode": "form", "requestedSchema": {"properties": {}}},
    )
    gateway.pending_server_requests["discord-thread"] = [pending_request]
    allow_action = gateway._server_request_actions(
        pending_request.method, pending_request.params
    )[0]
    gateway.client.respond_error = RuntimeError("socket unavailable")

    with pytest.raises(RuntimeError, match="socket unavailable"):
        await gateway._resolve_server_request_action(
            "discord-thread", pending_request, allow_action
        )

    assert gateway.pending_server_requests["discord-thread"][0] is pending_request
    gateway.client.respond_error = None
    assert await gateway._resolve_server_request_action(
        "discord-thread", pending_request, allow_action
    )


@pytest.mark.asyncio
async def test_handle_server_request_component_failure_sends_text_fallback(
    gateway: DiscordCodexGateway,
) -> None:
    """A failed interactive delivery sends the complete prompt as text."""
    gateway.bindings.bind(
        CodexTaskBinding("discord-thread", "codex-thread", "/tmp/project")
    )

    async def unavailable_components(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(success=False, error="components unavailable")

    gateway.adapter.send_codex_server_request = unavailable_components
    await gateway.handle_server_request(
        "approval-1",
        "item/fileChange/requestApproval",
        {"threadId": "codex-thread"},
    )

    assert gateway.adapter.sent == [
        (
            "discord-thread",
            "⚠️ **Codex requests file-change approval**\n"
            "Reply `/approve`, `/approve session`, or `/deny`.",
        )
    ]


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
async def test_resume_from_command_uses_raw_discord_name_guard(
    gateway: DiscordCodexGateway,
) -> None:
    """A first `/resume` mapping guards against the raw Discord thread name."""
    event = FakeMessageEvent("/resume Search result")
    event.source.chat_name = "Test server / #m7 / raw thread name"
    gateway.adapter.chat_names["discord-thread"] = "raw thread name"

    await gateway.handle_message(event)

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Mapped task | tmp/project",
            "raw thread name",
        )
    ]


@pytest.mark.asyncio
async def test_resume_from_command_uses_stored_discord_title_guard(
    gateway: DiscordCodexGateway,
) -> None:
    """Reattaching uses the Hermes-owned title guard instead of the live name."""
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "previous-thread",
            "/tmp/previous",
            title="Previous task",
            discord_title="✅ Previous task | tmp/previous | previous-thread",
        )
    )

    await gateway.handle_message(FakeMessageEvent("/resume Search result"))

    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Mapped task | tmp/project",
            "✅ Previous task | tmp/previous | previous-thread",
        )
    ]


@pytest.mark.asyncio
async def test_resume_without_selector_lists_recent_codex_tasks(
    gateway: DiscordCodexGateway,
) -> None:
    """The registered Discord resume command lists tasks when left empty."""
    response = await gateway.handle_message(FakeMessageEvent("/resume"))

    assert "Search result" in response
    assert gateway.client.requests[-1][0] == "thread/list"


@pytest.mark.asyncio
async def test_refresh_task_migrates_contextual_discord_title_guard(
    gateway: DiscordCodexGateway,
) -> None:
    """Refresh replaces a context-qualified guard after a successful rename."""
    contextual_guard = "Test server / #m7 / raw thread name"
    gateway.bindings.bind(
        CodexTaskBinding(
            "discord-thread",
            "codex-thread",
            "/tmp/original",
            title="Previous task",
            discord_title=contextual_guard,
        )
    )

    response = await gateway.handle_message(FakeMessageEvent("/refresh"))

    rendered_title = "✅ Mapped task | tmp/project"
    assert response.startswith("Refreshed Codex task `codex-thread`.")
    assert gateway.adapter.renamed_threads == [
        ("discord-thread", rendered_title, contextual_guard)
    ]
    assert gateway.bindings.bindings["discord-thread"].discord_title == (
        rendered_title
    )


@pytest.mark.asyncio
async def test_refresh_task_skips_history_replay(
    gateway: DiscordCodexGateway,
) -> None:
    """Refresh reads current status without reconstructing transcript messages."""
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
    assert gateway.adapter.sent == []
    assert all(
        method != "thread/turns/list" for method, _ in gateway.client.requests
    )
    assert gateway.bindings.bindings["discord-thread"].cwd == "/tmp/project"
    assert gateway.bindings.bindings["discord-thread"].title == "Mapped task"
    assert gateway.adapter.renamed_threads == [
        (
            "discord-thread",
            "✅ Mapped task | tmp/project",
            "Previous task",
        )
    ]
    assert gateway.bindings.bindings["discord-thread"].discord_title == (
        "✅ Mapped task | tmp/project"
    )


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
