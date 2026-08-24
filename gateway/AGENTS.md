# Codex Daemon Discord Gateway

## Scope

These instructions apply to the transparent Discord-to-Codex daemon gateway in
`gateway/codex_daemon_gateway.py` and its integration with `gateway/run.py`.
They extend the repository-wide instructions in the root `AGENTS.md` file.

The deployment model is one independent Hermes gateway on each compute server.
Each Hermes gateway connects to the Codex daemon on the same server and owns a
dedicated Discord bot and parent channel.

```text
m2 Codex daemon   <- local Unix socket -> m2 Hermes gateway   -> Discord #m2
m7 Codex daemon   <- local Unix socket -> m7 Hermes gateway   -> Discord #m7
vllm Codex daemon <- local Unix socket -> vllm Hermes gateway -> Discord #vllm
```

This design uses outbound Discord connections. Do not add SSH, Microsoft Dev
Tunnels, or a public relay to this deployment path.

## Behavioral contract

- Discord and `codex agents` must expose the same Codex task transcript.
- A Codex task belongs to the daemon on one server.
- One Discord thread represents one Codex task.
- Terminal-authored user messages and Codex responses must appear in Discord.
- Discord-authored messages must enter the attached Codex task.
- Tool activity may use a compact presentation, but the final response must
  remain a separate message.
- Restarting Hermes or the Codex daemon must preserve durable task-to-thread
  mappings and restore subscriptions.
- Each gateway must ignore Discord channels assigned to other servers.

## Source ownership

| File                              | Responsibility                                                    |
| --------------------------------- | ----------------------------------------------------------------- |
| `gateway/codex_daemon_gateway.py` | Connect to the local daemon and mirror Codex tasks into Discord.  |
| `gateway/run.py`                  | Start and stop the Codex gateway with the Discord adapter.        |
| `plugins/platforms/discord/`      | Deliver Discord messages, threads, files, and component controls. |
| `tests/gateway/`                  | Verify protocol, transcript, restart, and Discord behavior.       |

`CodexDaemonClient` owns the app-server JSON-RPC connection. It connects to
the local Unix socket with `aiohttp.UnixConnector`. `DiscordCodexGateway` owns
the mapping between Codex task identifiers and Discord thread identifiers.

Do not create a second transport or a second task-to-thread mapping store for
the per-server deployment model.

## Required inputs for one server

Collect these values before configuring a server:

- Server name, such as `m2`, `m7`, or `vllm`.
- Dedicated Discord bot token.
- Dedicated Discord parent-channel ID.
- Discord user ID for the operator.
- Absolute default working directory for Codex tasks.

Use a separate Discord bot token for each independently running gateway. A
shared token causes multiple gateway processes to receive the same Discord
events and can create duplicate command or message handling.

## Discord preparation

Create one Discord text channel for the server. Install the server's bot in the
Discord guild and grant the bot these permissions in that channel:

- View Channel
- Send Messages
- Read Message History
- Create Public Threads
- Send Messages in Threads
- Manage Threads
- Attach Files
- Use Application Commands

Enable Message Content Intent for the bot application. Restrict the bot's
channel permissions to its assigned server channel.

## Codex daemon setup

Run these commands as the account that owns the Codex tasks:

```bash
codex app-server daemon bootstrap
codex app-server daemon start
codex app-server daemon version
```

The standard daemon socket is:

```text
$HOME/.codex/app-server-control/app-server-control.sock
```

Hermes automatically runs `codex app-server daemon start` when this standard
socket is configured. The explicit commands remain part of installation and
diagnosis because they verify the managed Codex runtime before Hermes starts.

## Hermes installation

The `codex-gateway` branch must contain the tracked gateway source and tests
before another server clones it. Do not commit or push changes without explicit
user authorization.

Use an environment outside the source checkout:

```bash
git clone --branch codex-gateway \
  https://github.com/Davids048/hermes-agent.git "$HOME/repos/hermes-fork"

export HERMES_HOME="$HOME/.hermes-codex-gateway"
uv venv "$HERMES_HOME/venv" --python 3.11
source "$HERMES_HOME/venv/bin/activate"

cd "$HOME/repos/hermes-fork"
uv pip install -e ".[all]"
```

## Hermes secrets

Create `$HERMES_HOME/.env` with permissions `0600`:

```dotenv
DISCORD_BOT_TOKEN=SERVER_BOT_TOKEN
DISCORD_ALLOWED_USERS=OPERATOR_DISCORD_USER_ID
DISCORD_HOME_CHANNEL=SERVER_CHANNEL_ID
```

Never commit the `.env` file or print its values in logs, tests, or validation
artifacts.

## Hermes configuration

Create `$HERMES_HOME/config.yaml`:

```yaml
discord:
  allowed_channels:
    - "SERVER_CHANNEL_ID"
  auto_thread: true
  gateway_restart_notification: false
  require_mention: false

codex_gateway:
  enabled: true
  socket_path: "~/.codex/app-server-control/app-server-control.sock"
  default_cwd: "/ABSOLUTE/DEFAULT/WORKING/DIRECTORY"
  parent_chat_id: "SERVER_CHANNEL_ID"
  member_user_ids:
    - "OPERATOR_DISCORD_USER_ID"
```

`discord.allowed_channels` is the inbound channel boundary.
`codex_gateway.parent_chat_id` is the channel under which Hermes creates a
Discord thread for each Codex task. Set both fields to the same server-channel
ID.

## Persistent service

Install and start the Hermes gateway after the daemon and configuration are
ready:

```bash
export HERMES_HOME="$HOME/.hermes-codex-gateway"
source "$HERMES_HOME/venv/bin/activate"

hermes gateway install --start-now
hermes gateway status --deep
```

Use `hermes gateway restart` after changing the Hermes source or configuration.

## Deployment sequence

Deploy one server at a time:

1. Finish and validate the gateway implementation on the Mac checkout.
2. Obtain explicit authorization before staging, committing, or pushing.
3. Create the server's Discord bot and channel.
4. Bootstrap and verify the server's Codex daemon.
5. Clone the `codex-gateway` branch and install the Hermes environment.
6. Write the server's `.env` and `config.yaml` files.
7. Install the persistent Hermes gateway service.
8. Complete the acceptance checks before deploying the next server.

## Acceptance checks

For every server, verify all of these behaviors:

- `codex app-server daemon version` reports a running daemon.
- `hermes gateway status --deep` reports a running Discord gateway.
- A task created with `codex agents` appears as a Discord thread in the
  server's parent channel.
- Text typed in the Codex terminal appears in that Discord thread.
- The final Codex response appears in the same Discord thread.
- Text sent in Discord reaches the attached Codex task.
- A running turn streams progress without leaving and reopening the thread.
- Restarting Hermes restores the task-to-thread mappings.
- Restarting the Codex daemon allows Hermes to reconnect and resume mappings.
- The gateway does not respond in another server's Discord channel.

Run the focused repository validation before deployment:

```bash
scripts/run_tests.sh \
  tests/gateway/test_codex_daemon_gateway.py \
  tests/gateway/test_discord_send.py \
  tests/gateway/test_discord_edit_message_overflow.py \
  tests/gateway/test_discord_split_cap.py
```

Store validation logs under `archived/validation/codex-gateway/`.

## Source-control safeguards

- Preserve unrelated working-tree changes.
- Stage only the gateway source, affected Discord adapter code, focused tests,
  and this scoped `AGENTS.md` file when the user authorizes a commit.
- Do not stage `archived/` unless the user explicitly requests validation
  artifacts in version control.
- Do not commit credentials, daemon sockets, runtime state, or generated logs.
