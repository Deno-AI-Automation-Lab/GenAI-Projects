# n8n MCP Unread Email Summarizer

A chat-driven AI agent that uses n8n's **instance-level MCP server** to find and run another n8n workflow. That workflow fetches the last 5 unread Gmail messages, summarizes them with OpenAI, and emails the summary to the address given in the chat.

Example chat prompt:

```
summarize last 5 unread mails and send it to someone@example.com
```

---

## Architecture

```
┌──────────────────────────── Workflow 1: MCP N8N Instance level ────────────────────────────┐
│                                                                                            │
│  When chat message received ──► AI Agent (gpt-4.1-mini)                                    │
│                                   ├── Brain   (OpenAI Chat Model)                          │
│                                   ├── Memory  (Window Buffer Memory)                       │
│                                   └── MCP Client N8N Instance (tool)                       │
│                                          │  search_workflows / execute_workflow            │
└──────────────────────────────────────────┼─────────────────────────────────────────────────┘
                                           ▼
                         n8n instance MCP server  (/mcp-server/http, OAuth2)
                                           │
┌──────────────────────────────────────────┼── Workflow 2: MCP N8N Instance Unread Email Summarizer ──┐
│                                          ▼                                                          │
│  Webhook (POST /summarize-unread) ──► Get many messages (Gmail, UNREAD, limit 5)                    │
│  Trigger (Manual, for editor tests) ─┘        │                                                     │
│                                               ▼                                                     │
│                                   Aggregate (all items → "emails")                                  │
│                                               │                                                     │
│                                               ▼                                                     │
│                                   AI Agent (gpt-4.1-mini) — one-line summary per email              │
│                                               │                                                     │
│                                               ▼                                                     │
│                                   Send a message (Gmail) → recipient from webhook body              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Workflows

### 1. MCP N8N Instance level (client / chat agent)

| Node | Type | Purpose |
|---|---|---|
| When chat message received | Chat Trigger | Entry point for user prompts |
| AI Agent | LangChain Agent | Interprets the request, searches and runs workflows via MCP |
| Brain | OpenAI Chat Model (`gpt-4.1-mini`) | LLM for the agent |
| Memory | Window Buffer Memory | Keeps short chat history per session |
| MCP Client N8N Instance | MCP Client Tool | Connects to `https://<your-n8n-domain>/mcp-server/http` using MCP OAuth2 |

The agent's **system message** tells it to:
- Use `search_workflows` with up to 3 progressively broader searches, then `execute_workflow`
- Use any values already in the user's message (such as the email address) without asking for confirmation
- For the summarizer workflow, always use the **Webhook** trigger and pass the body `{"email": "<recipient>"}`

### 2. MCP N8N Instance Unread Email Summarizer (worker)

| Node | Type | Purpose |
|---|---|---|
| Webhook | Webhook (POST `summarize-unread`) | Trigger used by MCP `execute_workflow` |
| Trigger | Manual Trigger | For testing in the editor only (not usable over MCP) |
| Get many messages | Gmail – getAll | Fetches 5 messages with label `UNREAD` |
| Aggregate | Aggregate – all item data | Combines the 5 emails into one item (`emails`) so the LLM and Send run once |
| AI Agent | LangChain Agent | Produces exactly 5 one-line summaries (≤ 20 words each) |
| OpenAI Chat Model | `gpt-4.1-mini` | LLM for the summarizer |
| Send a message | Gmail – send | Emails the summary |

Recipient expression (falls back to a default address):

```
{{ $('Webhook').first().json.body.email || 'denomelvin24@gmail.com' }}
```

Workflow settings: **Published** and **Available in MCP = true**.

---

## Setup

See [REQUIREMENTS.md](REQUIREMENTS.md) for prerequisites.

1. **Import** both JSON files into n8n (*Workflows → Import from File*).
2. **Credentials** – attach or recreate:
   - `OpenAI account` on *Brain* and *OpenAI Chat Model*
   - `Gmail account` (Gmail OAuth2) on *Get many messages* and *Send a message*
   - `MCP N8N Instance` (MCP OAuth2) on *MCP Client N8N Instance*
3. **MCP endpoint** – in *MCP Client N8N Instance*, set the endpoint URL to your instance:
   `https://<your-n8n-domain>/mcp-server/http`
4. **Summarizer workflow**
   - Open *⋯ → Settings* and turn on **Available in MCP**
   - Update the fallback email in *Send a message* if needed
   - **Publish** the workflow
5. **Chat workflow** – **Publish** it.

## Usage

1. Open **MCP N8N Instance level** → *Open chat*.
2. Send: `summarize last 5 unread mails and send it to <email>`
3. Check the recipient's inbox and the **Executions** tab of the summarizer workflow.

Tip: reset the chat session (icon next to *Session*) after changing the agent's system message, so old memory doesn't affect new behaviour.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Agent says the workflow has no usable trigger / asks about "Manual Trigger" | The workflow only had a Manual Trigger, which MCP can't start | Add a **Webhook** trigger and republish |
| Agent asks to confirm the trigger or webhook data | System message missing trigger/body instructions | Keep the Webhook + body instruction in the agent's system message |
| Workflow not found by the agent | Not published, or *Available in MCP* is off | Publish the workflow and enable *Available in MCP* |
| Changes don't take effect | Edits to a published workflow aren't live until republished | Click **Publish** again |
| 5 separate summary emails | Missing Aggregate step | Keep **Aggregate → All Item Data** before the AI Agent |
| Gmail fails after ~7 days | Google OAuth app in **Testing** mode | Publish the OAuth app in Google Cloud Console, then reconnect the Gmail credential |
| Mail goes to the fallback address | Agent didn't send `body.email` | Check the execution's Webhook input; reinforce the system message |

## Security notes

- Never commit exported workflows that contain credential secrets. n8n exports only credential **IDs and names**, not the secrets themselves, but review files before sharing.
- The summarizer has a default recipient. Change it if others can use the chat.
- The Webhook node has no authentication. Add **Header Auth** if the production webhook URL may be exposed.
