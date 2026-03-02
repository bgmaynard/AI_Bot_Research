"""
Communications Bridge - File-based messaging between Claude and ChatGPT.

Architecture:
    Both AIs read/write JSON message files in the comms/ directory.
    The user syncs via git push/pull between sessions.

Message flow:
    Claude  -> comms/outbox_claude.json   -> git push -> git pull -> ChatGPT reads
    ChatGPT -> comms/outbox_chatgpt.json  -> git push -> git pull -> Claude reads

Usage (from either AI):
    from comms.bridge import CommsBridge

    bridge = CommsBridge()

    # Send a message
    bridge.send(
        from_agent="claude",
        to_agent="chatgpt",
        msg_type="status_update",
        subject="Phase 9 complete",
        body="Walk-forward validation passed. Results in reports/phase9.json",
        references=["reports/phase9.json", "Morpheus_Lab/strategies/flush_reclaim_v1.py"]
    )

    # Read incoming messages
    messages = bridge.receive(for_agent="claude")
    unread = bridge.get_unread(for_agent="claude")

    # Mark as read
    bridge.mark_read(msg_id="msg_20260302_001")
"""

import json
import os
from datetime import datetime, timezone


COMMS_DIR = os.path.dirname(os.path.abspath(__file__))

AGENTS = ("claude", "chatgpt")

MESSAGE_TYPES = (
    "status_update",    # Progress report, phase completion
    "question",         # Needs a response from the other agent
    "answer",           # Response to a question
    "data_handoff",     # Handing off data/files for the other to process
    "action_request",   # Requesting the other agent to do something
    "finding",          # Research finding or discovery to share
    "blocker",          # Something is blocked, needs attention
)


def _outbox_path(agent: str) -> str:
    """Return the outbox file path for a given agent."""
    return os.path.join(COMMS_DIR, f"outbox_{agent}.json")


def _load_outbox(agent: str) -> dict:
    """Load an agent's outbox, creating it if it doesn't exist."""
    path = _outbox_path(agent)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"agent": agent, "messages": []}


def _save_outbox(agent: str, data: dict) -> None:
    """Save an agent's outbox."""
    path = _outbox_path(agent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _next_msg_id(agent: str, existing_messages: list) -> str:
    """Generate the next message ID."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    count = sum(1 for m in existing_messages if m["id"].startswith(f"msg_{agent}_{date_str}"))
    return f"msg_{agent}_{date_str}_{count + 1:03d}"


class CommsBridge:
    """File-based communications bridge between AI agents."""

    def send(
        self,
        from_agent: str,
        to_agent: str,
        msg_type: str,
        subject: str,
        body: str,
        references: list[str] | None = None,
        in_reply_to: str | None = None,
    ) -> str:
        """
        Send a message by appending to the sender's outbox.

        Args:
            from_agent: Sender ("claude" or "chatgpt").
            to_agent: Recipient ("claude" or "chatgpt").
            msg_type: One of MESSAGE_TYPES.
            subject: Short subject line.
            body: Full message content.
            references: List of file paths relevant to this message.
            in_reply_to: Message ID this is responding to.

        Returns:
            The new message ID.
        """
        if from_agent not in AGENTS:
            raise ValueError(f"from_agent must be one of {AGENTS}, got '{from_agent}'")
        if to_agent not in AGENTS:
            raise ValueError(f"to_agent must be one of {AGENTS}, got '{to_agent}'")
        if msg_type not in MESSAGE_TYPES:
            raise ValueError(f"msg_type must be one of {MESSAGE_TYPES}, got '{msg_type}'")

        outbox = _load_outbox(from_agent)
        msg_id = _next_msg_id(from_agent, outbox["messages"])

        message = {
            "id": msg_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": from_agent,
            "to": to_agent,
            "type": msg_type,
            "subject": subject,
            "body": body,
            "references": references or [],
            "in_reply_to": in_reply_to,
            "read": False,
        }

        outbox["messages"].append(message)
        _save_outbox(from_agent, outbox)
        return msg_id

    def receive(self, for_agent: str) -> list[dict]:
        """
        Get all messages addressed to an agent (reads the OTHER agent's outbox).

        Args:
            for_agent: The agent who wants to read their incoming messages.

        Returns:
            List of messages addressed to for_agent.
        """
        messages = []
        for agent in AGENTS:
            if agent == for_agent:
                continue
            outbox = _load_outbox(agent)
            for msg in outbox["messages"]:
                if msg["to"] == for_agent:
                    messages.append(msg)
        return sorted(messages, key=lambda m: m["timestamp"])

    def get_unread(self, for_agent: str) -> list[dict]:
        """Get only unread messages for an agent."""
        return [m for m in self.receive(for_agent) if not m.get("read", False)]

    def mark_read(self, msg_id: str) -> bool:
        """
        Mark a message as read. Searches all outboxes.

        Returns:
            True if the message was found and marked, False otherwise.
        """
        for agent in AGENTS:
            outbox = _load_outbox(agent)
            for msg in outbox["messages"]:
                if msg["id"] == msg_id:
                    msg["read"] = True
                    _save_outbox(agent, outbox)
                    return True
        return False

    def get_conversation(self, between: tuple[str, str] | None = None) -> list[dict]:
        """
        Get all messages, optionally filtered to a conversation between two agents.

        Args:
            between: Tuple of two agent names, or None for all messages.

        Returns:
            All messages sorted by timestamp.
        """
        all_messages = []
        for agent in AGENTS:
            outbox = _load_outbox(agent)
            all_messages.extend(outbox["messages"])

        if between:
            a, b = between
            all_messages = [
                m for m in all_messages
                if (m["from"] == a and m["to"] == b)
                or (m["from"] == b and m["to"] == a)
            ]

        return sorted(all_messages, key=lambda m: m["timestamp"])

    def summary(self) -> dict:
        """Get a summary of all mailbox state."""
        result = {}
        for agent in AGENTS:
            outbox = _load_outbox(agent)
            sent = outbox["messages"]
            inbox = self.receive(agent)
            unread = self.get_unread(agent)
            result[agent] = {
                "sent": len(sent),
                "received": len(inbox),
                "unread": len(unread),
            }
        return result
