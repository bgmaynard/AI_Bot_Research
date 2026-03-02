# Communications Bridge Protocol
# Last Updated: 2026-03-02

## Purpose

File-based messaging system between Claude and ChatGPT for the Morpheus Lab
project. Neither AI can make outbound network calls, so this bridge uses git
as the transport layer.

## How It Works

```
Claude writes  -> comms/outbox_claude.json   -> git push
                                                   |
                                              git pull
                                                   |
ChatGPT reads  <- comms/outbox_claude.json   <- (user opens file)

ChatGPT writes -> comms/outbox_chatgpt.json  -> git push
                                                   |
                                              git pull
                                                   |
Claude reads   <- comms/outbox_chatgpt.json  <- (user opens file or Claude reads)
```

The user runs `git pull` / `git push` to sync messages between sessions.

## Message Schema

Each message in an outbox file follows this structure:

```json
{
  "id": "msg_claude_20260302_001",
  "timestamp": "2026-03-02T00:00:00+00:00",
  "from": "claude",
  "to": "chatgpt",
  "type": "status_update",
  "subject": "Short subject line",
  "body": "Full message content. Can be multiple paragraphs.",
  "references": ["path/to/relevant/file.py"],
  "in_reply_to": "msg_chatgpt_20260301_003",
  "read": false
}
```

## Message Types

| Type | Use Case |
|------|----------|
| status_update | Progress report, phase completion, what was built |
| question | Needs a response from the other agent |
| answer | Response to a previous question |
| data_handoff | Handing off data/files for the other to process |
| action_request | Requesting the other agent to do something specific |
| finding | Research finding or discovery worth sharing |
| blocker | Something is blocked, needs the other's input |

## For Claude (Claude Code)

Claude has direct file access via tools. Use the bridge utility:

```python
from comms.bridge import CommsBridge

bridge = CommsBridge()

# Send a message to ChatGPT
bridge.send(
    from_agent="claude",
    to_agent="chatgpt",
    msg_type="status_update",
    subject="Phase 9 results",
    body="Walk-forward validation complete. All details in the report.",
    references=["reports/phase9.json"]
)

# Read messages from ChatGPT
messages = bridge.receive(for_agent="claude")
unread = bridge.get_unread(for_agent="claude")

# Mark a message as read
bridge.mark_read("msg_chatgpt_20260302_001")
```

Or simply read/write the JSON files directly with the Read/Edit tools.

## For ChatGPT

ChatGPT operates in a sandbox and cannot call external APIs. The workflow is:

1. **To read Claude's messages**: Ask the user to paste the contents of
   `comms/outbox_claude.json`, or the user can upload the file directly.

2. **To send a message to Claude**: Generate the JSON message object and ask
   the user to append it to `comms/outbox_chatgpt.json`, then git commit + push.

Example message for ChatGPT to produce:

```json
{
  "id": "msg_chatgpt_20260302_001",
  "timestamp": "2026-03-02T12:00:00+00:00",
  "from": "chatgpt",
  "to": "claude",
  "type": "answer",
  "subject": "Re: Phase 9 results",
  "body": "Reviewed the walk-forward data. The OOS degradation pattern suggests...",
  "references": [],
  "in_reply_to": "msg_claude_20260302_001",
  "read": false
}
```

The user then pastes this into outbox_chatgpt.json and does `git add . && git commit && git push`.

## For the User (Sync Workflow)

### After a Claude Code session:
```bash
cd C:\AI_Bot_Research
git pull origin main
git add comms/
git commit -m "Sync comms: Claude outbox updated"
git push origin main
```

### Before a ChatGPT session:
Open `comms/outbox_claude.json` and paste its contents into ChatGPT,
or upload the file. Tell ChatGPT: "Here are the latest messages from Claude."

### After a ChatGPT session:
Copy ChatGPT's generated message JSON into `comms/outbox_chatgpt.json`,
then commit and push.

### Before a Claude Code session:
```bash
git pull origin main
```
Claude can then read `comms/outbox_chatgpt.json` directly.

## File Layout

```
comms/
  __init__.py              # Package init
  bridge.py                # Python utility (optional, for Claude's use)
  outbox_claude.json       # Claude's outgoing messages (ChatGPT reads this)
  outbox_chatgpt.json      # ChatGPT's outgoing messages (Claude reads this)
  BRIDGE_README.md         # This file
```

## Conventions

- Message IDs: `msg_{agent}_{YYYYMMDD}_{NNN}` (e.g., msg_claude_20260302_001)
- Timestamps: ISO 8601 UTC
- Keep messages concise -- reference files instead of inlining large data
- Use `in_reply_to` to thread conversations
- Mark messages as `"read": true` after processing them
