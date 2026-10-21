# Human in the Loop – Telegram Email Assistant (n8n)

A Telegram bot that drafts and sends Gmail emails through an AI agent. **No email goes out without human approval.** The Gmail tool is wrapped by n8n's **Telegram human-in-the-loop (HITL) gate**: before any email is sent, the bot posts the exact recipient, subject and body to Telegram with **✅ Send Email / ❌ Cancel Email** buttons, and the agent waits for a decision.

Example messages:

```
Send an email to someone@example.com saying the server patching is completed tonight
Draft an email to my team about tomorrow's maintenance window        ← drafted in chat only
What is the difference between SMTP and IMAP?                          ← answered normally
```

---

## Architecture

```
Telegram Trigger ──► AI Agent ──► Send a text message (Telegram reply)
                       │
                       ├── Brain   (OpenAI gpt-4.1-mini)
                       ├── Memory  (Window Buffer, session = Telegram chat ID)
                       │
                       └── Telegram (HITL approval gate)
                               │   posts approval request with ✅ / ❌ buttons
                               ▼
                           Action (Gmail – send message)
                               runs ONLY if approved
```

## Nodes

| Node | Type | Purpose |
|---|---|---|
| Telegram Trigger | Telegram Trigger (`message`) | Receives user messages |
| AI Agent | AI Agent v3.1 | Interprets the request: send, draft, or answer |
| Brain | OpenAI Chat Model `gpt-4.1-mini` | LLM for the agent |
| Memory | Window Buffer Memory | Per-chat memory; key `{{ $('Telegram Trigger').item.json.message.chat.id }}` |
| Telegram | Telegram HITL Tool | Approval gate in front of Gmail; sends an approval request to the same chat |
| Action | Gmail Tool – send | Sends the email; To, Subject and Message are filled by the agent via `$fromAI` |
| Send a text message | Telegram – send | Replies with the agent's final answer |

### Approval message

```
The agent wants to run this tool: {{ $tool.name }}

Input:
{{ JSON.stringify($tool.parameters, null, 2) }}
```

- Approval type: **double** (two buttons)
- Approve label: `✅ Send Email`
- Disapprove label: `❌ Cancel Email`

## Agent behaviour (system message)

| User request | Behaviour |
|---|---|
| "Send / email …" | Extracts recipient, subject (creates a short one if missing) and message, then calls Gmail through the approval gate |
| "Compose / draft …" without "send" | Drafts in chat only; Gmail is not used |
| General question | Answers normally |
| Approved | Gmail runs; the bot confirms sent only after Gmail succeeds |
| Rejected | Treated as final: reports "cancelled, not sent", doesn't retry or ask again |

Accuracy rules: never invent recipient addresses, approval status, or delivery results.

---

## Setup

See [REQUIREMENTS.md](REQUIREMENTS.md) for prerequisites.

1. **Telegram bot**: create one with @BotFather and copy the token. Use a bot token that **isn't** used by another active workflow.
2. **Import** `Human in the Loop.json` into n8n.
3. **Credentials**:
   - `Telegram account` on Telegram Trigger, Telegram (HITL), and Send a text message
   - `OpenAI account` on Brain
   - `Gmail account` (Gmail OAuth2) on Action
4. **Publish** the workflow. This registers the Telegram webhook and the approval callback.

## Usage

1. Message the bot: `Send an email to x@example.com about …`
2. The bot posts an approval request showing the exact email parameters.
3. Tap **✅ Send Email** to send, or **❌ Cancel Email** to cancel.
4. The bot replies with the result.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Error in sub-node "Memory": *Key parameter is empty* | Session ID set to Connected Chat Trigger, or the expression pasted into the Session ID dropdown | Session ID → **Define below**, then put the expression in **Key** |
| No approval message appears | Workflow not published, or HITL node chat ID wrong | Publish; check the HITL `chatId` expression |
| Approval buttons do nothing | Execution expired, or the webhook can't reach n8n | Check `WEBHOOK_URL` / HTTPS; retry with a new request |
| Bot sends the email again after Cancel | Prompt not followed | Keep the "rejection is final" rules in the system message; reset memory |
| Gmail fails after ~7 days | Google OAuth app in Testing mode | Publish the OAuth app, then reconnect the Gmail credential |
| Telegram webhook conflict | Same bot token used by another workflow | One active Telegram Trigger per bot token |

## Notes and possible improvements

- **Approver = requester.** The approval request goes to the same chat that made the request, so the user approves their own emails. That's a good safety check against agent mistakes. For true four-eyes control, set the HITL `chatId` to a supervisor's chat ID or a group.
- Restrict who can use the bot by adding an **IF** node after the trigger that allows only known Telegram chat IDs.
- The Action tool description starts with "Send a message in GmailUse this tool…". Adding a space or line break after "Gmail" makes it read cleanly for the LLM.
