"""Tests for Discord controls that answer Codex app-server requests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.codex_daemon_gateway import CodexDiscordAction
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import (
    CodexServerRequestView,
    DiscordAdapter,
)


def _make_adapter() -> DiscordAdapter:
    """Build a connected Discord adapter with one authorized account."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={}))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    return adapter


def _make_interaction(user_id: str = "42") -> SimpleNamespace:
    """Build the Discord interaction fields consumed by the Codex view."""
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="David", roles=[]),
        message=SimpleNamespace(content="Codex needs approval."),
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
    )


@pytest.mark.asyncio
async def test_send_codex_server_request_attaches_action_view() -> None:
    """The adapter places mapped Codex actions on the prompt message."""
    adapter = _make_adapter()
    channel = MagicMock()
    sent_message = MagicMock(id=123)
    channel.send = AsyncMock(return_value=sent_message)
    adapter._client.get_channel.return_value = channel
    actions = [
        CodexDiscordAction("Allow", "answer", "{}", "success"),
        CodexDiscordAction("Deny", "approval", "deny", "danger"),
    ]
    on_action = AsyncMock(return_value=True)

    result = await adapter.send_codex_server_request(
        chat_id="9001",
        prompt="Allow Computer Use to use TickTick?",
        actions=actions,
        on_action=on_action,
    )

    assert result.success is True
    sent_view = channel.send.await_args.kwargs["view"]
    assert isinstance(sent_view, CodexServerRequestView)
    assert [button.label for button in sent_view.children] == ["Allow", "Deny"]


@pytest.mark.asyncio
async def test_send_codex_server_request_preserves_oversized_prompt() -> None:
    """The adapter sends a long prompt before its separate component row."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=MagicMock(id=123))
    adapter._client.get_channel.return_value = channel
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    action = CodexDiscordAction("Allow", "answer", "{}", "success")
    prompt = "x" * (adapter.MAX_MESSAGE_LENGTH + 1)

    result = await adapter.send_codex_server_request(
        chat_id="9001",
        prompt=prompt,
        actions=[action],
        on_action=AsyncMock(return_value=True),
    )

    assert result.success is True
    adapter.send.assert_awaited_once_with("9001", prompt)
    assert channel.send.await_args.kwargs["content"] == (
        "Choose a response for the Codex request."
    )


@pytest.mark.asyncio
async def test_send_codex_server_request_control_failure_preserves_long_prompt() -> None:
    """A delivered long prompt remains usable when its component row fails."""
    adapter = _make_adapter()
    channel = MagicMock()
    channel.send = AsyncMock(side_effect=RuntimeError("components unavailable"))
    adapter._client.get_channel.return_value = channel
    prompt_result = SimpleNamespace(success=True, message_id="456")
    adapter.send = AsyncMock(return_value=prompt_result)
    action = CodexDiscordAction("Allow", "answer", "{}", "success")
    prompt = "x" * (adapter.MAX_MESSAGE_LENGTH + 1)

    result = await adapter.send_codex_server_request(
        chat_id="9001",
        prompt=prompt,
        actions=[action],
        on_action=AsyncMock(return_value=True),
    )

    assert result is prompt_result
    adapter.send.assert_awaited_once_with("9001", prompt)


@pytest.mark.asyncio
async def test_codex_server_request_view_authorized_click_submits_once() -> None:
    """An authorized click resolves Codex and disables every response button."""
    action = CodexDiscordAction("Allow", "answer", "{}", "success")
    on_action = AsyncMock(return_value=True)
    view = CodexServerRequestView(
        actions=[action],
        on_action=on_action,
        allowed_user_ids={"42"},
    )
    interaction = _make_interaction()

    await view._resolve(interaction, action)

    on_action.assert_awaited_once_with(action)
    assert view.resolved is True
    assert all(button.disabled for button in view.children)
    edited_content = interaction.response.edit_message.await_args.kwargs["content"]
    assert "✅ **Allow** by David" in edited_content
    second_interaction = _make_interaction()
    await view._resolve(second_interaction, action)
    on_action.assert_awaited_once_with(action)


@pytest.mark.asyncio
async def test_codex_server_request_view_unauthorized_click_is_rejected() -> None:
    """A Discord account outside the allowlist cannot answer Codex prompts."""
    action = CodexDiscordAction("Allow", "answer", "{}", "success")
    on_action = AsyncMock(return_value=True)
    view = CodexServerRequestView(
        actions=[action],
        on_action=on_action,
        allowed_user_ids={"42"},
    )
    interaction = _make_interaction(user_id="99")

    await view._resolve(interaction, action)

    on_action.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
