import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
import plugins.platforms.discord.adapter as discord_platform  # noqa: E402


def _capture_discord_text_file(monkeypatch):
    """Replace discord.File with a readable in-memory attachment record."""
    created_files = []

    def build_file(fp, filename=None, **_kwargs):
        captured_file = SimpleNamespace(fp=fp, filename=filename)
        created_files.append(captured_file)
        return captured_file

    monkeypatch.setattr(discord_platform.discord, "File", build_file)
    return created_files


@pytest.mark.asyncio
async def test_send_oversized_fenced_code_block_uses_text_attachment(
    monkeypatch,
):
    """An oversized code block stays atomic between surrounding prose."""
    created_files = _capture_discord_text_file(monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent_messages = []

    async def send_message(**kwargs):
        sent_messages.append(kwargs)
        return SimpleNamespace(id=800 + len(sent_messages))

    channel = SimpleNamespace(send=AsyncMock(side_effect=send_message))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    code = "line\n" * 420

    result = await adapter.send(
        "555",
        f"Before diagram.\n\n```text\n{code}```\n\nAfter diagram.",
    )

    assert result.success is True
    assert [message["content"] for message in sent_messages] == [
        "Before diagram.",
        "📎 **Code block attached:** `code-block-1.txt`",
        "After diagram.",
    ]
    assert "files" not in sent_messages[0]
    assert sent_messages[1]["files"] == [created_files[0]]
    assert created_files[0].filename == "code-block-1.txt"
    assert created_files[0].fp.getvalue() == code.encode("utf-8")
    assert "files" not in sent_messages[2]


@pytest.mark.asyncio
async def test_send_fenced_code_block_at_message_limit_stays_inline():
    """A fenced block that fits Discord's limit remains normal Markdown."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id=888)),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    fenced_code = f"```\n{'x' * 1992}\n```"

    result = await adapter.send("555", fenced_code)

    assert len(fenced_code) == adapter.MAX_MESSAGE_LENGTH
    assert result.success is True
    call = channel.send.await_args.kwargs
    assert call["content"] == fenced_code
    assert "files" not in call


@pytest.mark.asyncio
async def test_edit_message_oversized_fenced_code_block_uses_text_attachment(
    monkeypatch,
):
    """A streamed answer finalizes into one attachment on its original message."""
    created_files = _capture_discord_text_file(monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(
        get_partial_message=MagicMock(return_value=message),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    code = "diagram row\n" * 190

    result = await adapter.edit_message(
        "555",
        "777",
        f"```text\n{code}```",
        finalize=True,
    )

    assert result.success is True
    assert result.message_id == "777"
    edit = message.edit.await_args.kwargs
    assert edit["content"] == "📎 **Code block attached:** `code-block-1.txt`"
    assert edit["attachments"] == [created_files[0]]
    assert created_files[0].fp.getvalue() == code.encode("utf-8")
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_codex_user_prompt_uses_visible_quoted_text():
    """Terminal-authored Codex input is visible without Discord embeds."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id=888)),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send(
        "555",
        "terminal input",
        metadata={"message_style": "codex_user_prompt"},
    )

    assert result.success is True
    call = channel.send.await_args.kwargs
    assert call["content"] == "**👤 You (Codex terminal)**\n>>> terminal input"
    assert "embed" not in call


@pytest.mark.asyncio
async def test_edit_message_codex_user_prompt_replaces_placeholder_with_text():
    """A working placeholder becomes separate visible terminal-input text."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(edit=AsyncMock())
    channel = SimpleNamespace(
        get_partial_message=MagicMock(return_value=message),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    adapter.register_codex_activity("777", "working", "command output")

    result = await adapter.edit_message(
        "555",
        "777",
        "terminal input",
        finalize=True,
        metadata={"message_style": "codex_user_prompt"},
    )

    assert result.success is True
    call = message.edit.await_args.kwargs
    assert call["content"] == "**👤 You (Codex terminal)**\n>>> terminal input"
    assert call["embed"] is None
    assert call["view"] is None
    assert "777" not in adapter._codex_activity_presentations
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_codex_turn_activity_adds_collapsed_controls(monkeypatch):
    """A Codex activity message starts collapsed with persistent controls."""
    monkeypatch.setattr(
        discord_platform,
        "CodexActivityView",
        lambda *args, **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(
        send=AsyncMock(return_value=SimpleNamespace(id=888)),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send(
        "555",
        "⏳ **Activity · 2 steps**\n1 command · 1 update",
        metadata={
            "message_style": "codex_turn_activity",
            "expanded_content": "### Codex activity\n\ncommand details",
        },
    )

    assert result.success is True
    call = channel.send.await_args.kwargs
    assert call["content"] == "⏳ **Activity · 2 steps**\n1 command · 1 update"
    assert call["view"].kwargs["expanded"] is False
    presentation = adapter._codex_activity_presentations["888"]
    assert presentation.expanded_pages == [
        "### Codex activity\n\ncommand details"
    ]


@pytest.mark.asyncio
async def test_handle_codex_activity_interaction_pages_and_collapses(monkeypatch):
    """Activity controls reveal every page and restore the compact summary."""
    monkeypatch.setattr(
        discord_platform,
        "CodexActivityView",
        lambda *args, **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"123"}
    adapter.register_codex_activity("888", "collapsed", "x" * 2500)
    response = SimpleNamespace(
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123, roles=[]),
        message=SimpleNamespace(id=888),
        response=response,
    )

    await adapter._handle_codex_activity_interaction(interaction, "show")
    first_page = response.edit_message.await_args.kwargs
    assert len(first_page["content"]) <= adapter.MAX_MESSAGE_LENGTH
    assert first_page["view"].kwargs["expanded"] is True

    await adapter._handle_codex_activity_interaction(interaction, "next")
    second_page = response.edit_message.await_args.kwargs
    assert second_page["content"] != first_page["content"]
    assert second_page["view"].kwargs["page_index"] == 1

    await adapter._handle_codex_activity_interaction(interaction, "hide")
    collapsed = response.edit_message.await_args.kwargs
    assert collapsed["content"] == "collapsed"
    assert collapsed["view"].kwargs["expanded"] is False


@pytest.mark.asyncio
async def test_handle_codex_activity_interaction_waits_without_notification(
    monkeypatch,
):
    """A restart-time click waits for history without posting a Discord message."""
    monkeypatch.setattr(
        discord_platform,
        "CodexActivityView",
        lambda *args, **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"123"}
    deferred = asyncio.Event()

    async def mark_deferred():
        deferred.set()

    response = SimpleNamespace(
        defer=AsyncMock(side_effect=mark_deferred),
        edit_message=AsyncMock(),
        send_message=AsyncMock(),
    )
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123, roles=[]),
        message=SimpleNamespace(id=888),
        response=response,
        edit_original_response=AsyncMock(),
    )

    interaction_task = asyncio.create_task(
        adapter._handle_codex_activity_interaction(interaction, "show")
    )
    await deferred.wait()
    adapter.register_codex_activity("888", "collapsed", "expanded details")
    await interaction_task

    response.send_message.assert_not_awaited()
    response.edit_message.assert_not_awaited()
    interaction.edit_original_response.assert_awaited_once()
    edit = interaction.edit_original_response.await_args.kwargs
    assert edit["content"] == "expanded details"
    assert edit["view"].kwargs["expanded"] is True


@pytest.mark.asyncio
async def test_create_codex_task_thread_posts_visible_starter_message(
    monkeypatch, tmp_path
):
    """A daemon task thread starts from a visible parent-channel message."""
    class FakeTextChannel:
        """Represent a Discord text channel accepted by the adapter."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        discord_platform.discord,
        "Object",
        lambda *, id: SimpleNamespace(id=id),
    )
    disabled_mentions = object()
    monkeypatch.setattr(
        discord_platform.discord,
        "AllowedMentions",
        SimpleNamespace(none=lambda: disabled_mentions),
    )
    monkeypatch.setattr(discord_platform.discord, "TextChannel", FakeTextChannel)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    created_thread = SimpleNamespace(id=777, add_user=AsyncMock())
    seed_message = SimpleNamespace(
        create_thread=AsyncMock(return_value=created_thread),
    )
    parent = FakeTextChannel()
    parent.send = AsyncMock(return_value=seed_message)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: parent,
        fetch_channel=AsyncMock(),
    )

    thread_id = await adapter.create_codex_task_thread(
        "555",
        "Terminal task",
        task_title="Inspect the repository",
        directory="codes/hermes-agent",
        member_user_ids=("123",),
    )

    assert thread_id == "777"
    assert "777" in adapter._threads
    parent.send.assert_awaited_once()
    send_args = parent.send.await_args
    assert send_args.args == (
        "🧵 Codex task: Inspect the repository\n"
        "📁 Directory: codes/hermes-agent",
    )
    assert send_args.kwargs["allowed_mentions"] is disabled_mentions
    seed_message.create_thread.assert_awaited_once_with(
        name="Terminal task",
        auto_archive_duration=1440,
        reason="Codex task created by another client",
    )
    created_thread.add_user.assert_awaited_once()
    assert created_thread.add_user.await_args.args[0].id == 123


def test_codex_task_starter_message_normalizes_and_bounds_metadata(
    monkeypatch, tmp_path
):
    """The visible marker fits Discord after normalizing task metadata."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    content = adapter._codex_task_starter_message(
        "Inspect\n" + "x" * 2500,
        "codes /  hermes-agent",
    )

    assert len(content) == adapter.MAX_MESSAGE_LENGTH
    assert content.startswith("🧵 Codex task: Inspect x")
    assert content.endswith("\n📁 Directory: codes / hermes-agent")


def test_codex_task_starter_message_escapes_discord_syntax(
    monkeypatch, tmp_path
):
    """Task metadata displays literally without formatting or live mentions."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    content = adapter._codex_task_starter_message(
        "**Review** <@12345678901234567> `<#12345678901234567>`",
        "codes/a_b",
    )

    assert content == (
        "🧵 Codex task: \\*\\*Review\\*\\* "
        "\\<@\u200b12345678901234567\\> "
        "\\`\\<#12345678901234567\\>\\`\n"
        "📁 Directory: codes/a\\_b"
    )


@pytest.mark.asyncio
async def test_create_codex_task_thread_rejects_non_text_parent(
    monkeypatch, tmp_path
):
    """An unsupported parent cannot receive an orphan Codex task marker."""
    class FakeTextChannel:
        """Represent the only parent type that can own a message thread."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(discord_platform.discord, "TextChannel", FakeTextChannel)
    parent = SimpleNamespace(send=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: parent,
        fetch_channel=AsyncMock(),
    )

    thread_id = await adapter.create_codex_task_thread(
        "555",
        "Terminal task",
        task_title="Inspect the repository",
        directory="codes/hermes-agent",
    )

    assert thread_id is None
    parent.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_rejects_whitespace_and_records_failed_final_reply(
    caplog, monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(send=AsyncMock())
    get_channel = MagicMock(return_value=channel)
    adapter._client = SimpleNamespace(
        get_channel=get_channel,
        fetch_channel=AsyncMock(),
    )
    with caplog.at_level("WARNING"):
        result = await adapter.send(
            "555",
            "  \n\t ",
            reply_to="123",
            metadata={"notify": True},
        )

    assert result.success is False
    assert result.error == "Refusing to send empty message"
    get_channel.assert_not_called()
    channel.send.assert_not_awaited()
    row = adapter._with_discord_recovery_db(
        lambda conn: conn.execute(
            "SELECT status, replied, outage_response, response_message_id "
            "FROM discord_messages WHERE message_id='123'"
        ).fetchone()
    )
    assert tuple(row) == ("failed", 0, 0, None)
    assert "Dropped empty message to chat=555" in caplog.text


def _voice_adapter(reference_obj, *, native_result=None, native_error=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(return_value=SimpleNamespace(id=888)),
    )
    request = AsyncMock(return_value=native_result or {"id": "777"})
    if native_error is not None:
        request.side_effect = native_error
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=request),
    )
    return adapter, channel, request


def _native_voice_payload(request):
    form = request.await_args.kwargs["form"]
    payload = next(part["value"] for part in form if part["name"] == "payload_json")
    return json.loads(payload)


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_deleted():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    sent_msgs = [SimpleNamespace(id=1001), SimpleNamespace(id=1002)]
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 10008): Unknown Message"
            )
        return sent_msgs[len(send_calls) - 2]

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    long_text = "A" * (adapter.MAX_MESSAGE_LENGTH + 50)
    result = await adapter.send("555", long_text, reply_to="99")

    assert result.success is True
    assert result.message_id == "1001"
    # ids-only reference: the fetch is gone entirely — the retry happens
    # on the send-side 10008, not a fetch failure
    assert channel.fetch_message.await_count == 0
    assert channel.send.await_count == 3
    # the reference is constructed from ids, not fetched + to_reference()
    _discord_mod.MessageReference.assert_any_call(
        message_id=99, channel_id=None, guild_id=None,
        fail_if_not_exists=False)
    assert send_calls[0]["reference"] is _discord_mod.MessageReference.return_value
    assert send_calls[1]["reference"] is None
    assert send_calls[2]["reference"] is None


# ---------------------------------------------------------------------------
# Forum channel tests
# ---------------------------------------------------------------------------

import discord as _discord_mod  # noqa: E402 — imported after _ensure_discord_mock


class TestIsForumParent:
    def test_none_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        assert adapter._is_forum_parent(None) is False

    def test_forum_channel_class_instance(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        forum_cls = getattr(_discord_mod, "ForumChannel", None)
        if forum_cls is None:
            # Re-create a type for the mock
            forum_cls = type("ForumChannel", (), {})
            _discord_mod.ForumChannel = forum_cls
        ch = forum_cls()
        assert adapter._is_forum_parent(ch) is True


# ---------------------------------------------------------------------------
# Forum follow-up chunk failure reporting + media on forum paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forum_post_file_creates_thread_with_attachment():
    """_forum_post_file routes file-bearing sends to create_thread with file kwarg."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread_ch = SimpleNamespace(id=777, send=AsyncMock())
    thread = SimpleNamespace(
        id=777,
        message=SimpleNamespace(
            id=800,
            attachments=[SimpleNamespace(filename="photo.png")],
        ),
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(return_value=thread)

    # discord.File is a real class; build a MagicMock that looks like one
    fake_file = SimpleNamespace(filename="photo.png")

    result = await adapter._forum_post_file(
        forum_channel,
        content="here is a photo",
        file=fake_file,
    )

    assert result.success is True
    assert result.message_id == "800"
    forum_channel.create_thread.assert_awaited_once()
    call_kwargs = forum_channel.create_thread.await_args.kwargs
    assert call_kwargs["file"] is fake_file
    assert call_kwargs["content"] == "here is a photo"
    # Thread name derived from content's first line
    assert call_kwargs["name"] == "here is a photo"


@pytest.mark.asyncio
async def test_forum_post_file_fails_when_starter_has_no_attachments():
    """Forum create_thread can succeed yet return an attachmentless starter (#66797)."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread = SimpleNamespace(
        id=7,
        message=SimpleNamespace(id=8, attachments=[]),
        thread=SimpleNamespace(id=7, send=AsyncMock()),
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.create_thread = AsyncMock(return_value=thread)

    fake_file = SimpleNamespace(filename="clip.mp4")
    result = await adapter._forum_post_file(
        forum_channel,
        content="video clip",
        files=[fake_file],
    )

    assert result.success is False
    assert "no files" in (result.error or "").lower()
    forum_channel.create_thread.assert_awaited_once()


# ---------------------------------------------------------------------------
# Typing indicator task lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_restartable_after_error():
    """After a typing error, send_typing should start a new task (not blocked by stale entry)."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._typing_tasks = {}

    # First call fails
    adapter._client.http.request = AsyncMock(side_effect=Exception("503"))
    await adapter.send_typing("12345")
    await asyncio.sleep(0.1)

    # Second call should work
    adapter._client.http.request = AsyncMock()
    await adapter.send_typing("12345")

    assert "12345" in adapter._typing_tasks, \
        "Should restart typing after previous failure"


# ---------------------------------------------------------------------------
# #66797 — outbound MEDIA video must reach channel.send as a real attachment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_video_uses_path_based_files_kwarg(tmp_path, monkeypatch):
    """Regression for #66797: video MEDIA delivery must use path-based
    ``discord.File`` via ``files=[...]`` (same pattern as image batching).

    The previous open-handle + singular ``file=`` form could return a successful
    message with zero attachments after an earlier image batch on the same
    channel — silent drop from the user's perspective.
    """
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    captured = {}

    class _FakeFile:
        def __init__(self, fp, filename=None, **kwargs):
            captured["fp"] = fp
            captured["filename"] = filename

    monkeypatch.setattr(discord_platform.discord, "File", _FakeFile)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent_msg = SimpleNamespace(
        id=4242,
        attachments=[SimpleNamespace(filename="clip.mp4", url="https://cdn.example/clip.mp4")],
    )
    channel = SimpleNamespace(
        send=AsyncMock(return_value=sent_msg),
        type=0,
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: False)

    result = await adapter.send_video("555", str(video))

    assert result.success is True
    assert result.message_id == "4242"
    assert captured["fp"] == str(video)
    assert captured["filename"] == "clip.mp4"
    channel.send.assert_awaited_once()
    send_kwargs = channel.send.await_args.kwargs
    assert send_kwargs.get("file") is None
    assert isinstance(send_kwargs.get("files"), list) and len(send_kwargs["files"]) == 1


@pytest.mark.asyncio
async def test_send_video_fails_loud_when_message_has_no_attachments(tmp_path, monkeypatch):
    """If Discord accepts the message but attaches nothing, fail loud (#66797)."""
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    monkeypatch.setattr(
        discord_platform.discord,
        "File",
        lambda fp, filename=None, **kwargs: SimpleNamespace(fp=fp, filename=filename),
    )

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    # Message id present, but no attachments — the silent-drop failure mode.
    sent_msg = SimpleNamespace(id=99, attachments=[])
    channel = SimpleNamespace(send=AsyncMock(return_value=sent_msg), type=0)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: False)

    result = await adapter.send_video("555", str(video))

    assert result.success is False
    assert "no files" in (result.error or "").lower()
    channel.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_video_missing_file_fails_fast_without_touching_channel():
    """A missing MEDIA path must fail loud before any Discord I/O (#66797).

    The pre-flight ``os.path.isfile`` guard turns a would-be crash inside
    ``discord.File`` into an actionable ``File not found`` result, and must
    short-circuit before the channel is ever resolved.
    """
    def _boom(*_args, **_kwargs):
        raise AssertionError("channel must not be resolved for a missing file")

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(get_channel=_boom, fetch_channel=AsyncMock(side_effect=_boom))

    result = await adapter.send_video("555", "/no/such/clip.mp4")

    assert result.success is False
    assert "not found" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_send_file_attachment_forum_uses_files_kwarg(tmp_path, monkeypatch):
    """Forum-parent delivery must also route the path-based file through the
    plural ``files=[...]`` kwarg (#66797), so the create_thread starter message
    carries the attachment rather than silently dropping it."""
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    monkeypatch.setattr(
        discord_platform.discord,
        "File",
        lambda fp, filename=None, **kwargs: SimpleNamespace(fp=fp, filename=filename),
    )

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    created_thread = SimpleNamespace(
        id=7,
        message=SimpleNamespace(
            id=8,
            attachments=[SimpleNamespace(filename="clip.mp4")],
        ),
    )
    forum_channel = SimpleNamespace(
        id=7,
        create_thread=AsyncMock(return_value=created_thread),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: True)

    result = await adapter.send_video("555", str(video))

    assert result.success is True
    forum_channel.create_thread.assert_awaited_once()
    thread_kwargs = forum_channel.create_thread.await_args.kwargs
    assert thread_kwargs.get("file") is None
    assert isinstance(thread_kwargs.get("files"), list) and len(thread_kwargs["files"]) == 1
