# n8n MCP Server + AI Agent Client (Multi-Tool Automation)

An agentic automation project built on **n8n** that exposes Google Workspace, web search and a calculator as **Model Context Protocol (MCP)** tools, and consumes them from a conversational **AI Agent** powered by OpenAI `gpt-4.1-mini`.

The project is split into two n8n workflows:

| Workflow | File | Role |
|---|---|---|
| **MCP Server Trigger** | `MCP Server Trigger.json` | MCP **server**: publishes 6 tools over an OAuth2-protected MCP endpoint |
| **Basic MCP Server Client Multi Tool** | `Basic MCP Server Client Multi Tool.json` | MCP **client**: chat-driven AI Agent that discovers and calls those tools |

Because the tools live behind an MCP endpoint, the same server can also be used by any other MCP-compatible client (Claude Desktop, Cursor, other n8n agents, custom LangChain apps).

---

## Architecture

```
 ┌──────────────────────────── Client workflow ─────────────────────────────┐
 │                                                                          │
 │  [When chat message received] ──► [Perception: AI Agent]                 │
 │                                        ▲        ▲         ▲              │
 │                         ai_languageModel│  ai_memory│   ai_tool│         │
 │                                        │        │         │              │
 │                         [Brain: OpenAI  ]  [Memory:    ]  [MCP Client]   │
 │                         [gpt-4.1-mini   ]  [Buffer Win.]       │         │
 └────────────────────────────────────────────────────────────────┼─────────┘
                                                                  │ SSE + OAuth2
 ┌──────────────────────────── Server workflow ───────────────────┼─────────┐
 │                                                                ▼         │
 │                           [MCP Server Trigger]  /mcp/<path-id>           │
 │                                     ▲                                    │
 │      ┌────────────┬─────────────┬───┴────────┬────────────┬──────────┐   │
 │  Google Docs  Google Calendar  Google Sheets   Gmail      SerpApi  Calculator
 │  (create doc) (create event)  (append row)  (send mail) (web search)     │
 └──────────────────────────────────────────────────────────────────────────┘
```

**Agent model (Perception → Brain → Memory → Tools)**

- **Perception** – n8n AI Agent node that interprets the user's chat message and decides whether a tool is needed.
- **Brain** – OpenAI Chat Model (`gpt-4.1-mini`) doing the reasoning and tool selection.
- **Memory** – Window Buffer Memory keeping recent turns of the conversation.
- **MCP Client** – connects to the MCP server and makes the 6 tools available to the agent.

---

## Tools exposed by the MCP Server

All tool parameters are filled by the LLM at runtime using n8n's `$fromAI()` expressions.

| MCP tool name | n8n node | What it does | AI-filled parameters |
|---|---|---|---|
| `Create_a_document_in_Google_Docs` | Google Docs Tool | Creates a new Google Doc in a fixed Drive folder | `Title` |
| `Create_an_event_in_Google_Calendar` | Google Calendar Tool | Creates an event on the configured calendar | `Start`, `End`, `Summary`, `Description` |
| `Append_row_in_sheet_in_Google_Sheets` | Google Sheets Tool | Appends a row to the `SalesData` sheet | `Date`, `Customer`, `Product`, `Quantity`, `Amount`, `Status` |
| `Send_a_message_in_Gmail` | Gmail Tool | Sends an email (n8n attribution disabled) | `To`, `Subject`, `Message` |
| `Google_search_in_SerpApi` | SerpApi Tool | Google web search for current information | `q` (query), `location` |
| `Calculator` | Calculator Tool | Evaluates arithmetic expressions | expression |

---

## Prerequisites

See **[REQUIREMENTS.md](REQUIREMENTS.md)** for the full list. In short:

- A self-hosted n8n instance reachable over **HTTPS** (this project runs on a Hostinger VPS)
- n8n version with the MCP Server Trigger, MCP Client Tool and AI Agent v3 nodes
- The `n8n-nodes-serpapi` community node installed
- Google Cloud OAuth client (Docs, Calendar, Sheets, Gmail, Drive APIs enabled)
- OpenAI API key and SerpApi API key

---

## Setup

### 1. Import the workflows
In n8n: **Workflows → Import from File**, and import both JSON files.

### 2. Create credentials
Create these in **Credentials** and re-select them on each node (credential IDs in the JSON are instance-specific and will not match your instance):

| Credential type | Used by |
|---|---|
| Google Docs OAuth2 API | Create a document in Google Docs |
| Google Calendar OAuth2 API | Create an event in Google Calendar |
| Google Sheets OAuth2 API | Append row in sheet in Google Sheets |
| Gmail OAuth2 | Send a message in Gmail |
| SerpApi | Google search in SerpApi |
| OpenAI API | Brain (OpenAI Chat Model) |
| MCP OAuth2 API | MCP Client |

For Google OAuth, add your n8n OAuth callback URL (`https://<your-n8n-host>/rest/oauth2-credential/callback`) to the Google Cloud OAuth client's authorised redirect URIs.

### 3. Point the server tools at your own resources
In the **MCP Server Trigger** workflow, update:

- **Google Docs** → `folderId`: the Drive folder where documents should be created
- **Google Calendar** → `calendar`: your calendar ID / email
- **Google Sheets** → `documentId` and `sheetName`: your spreadsheet and sheet. The sheet must have a header row with at least:
  `Date | Customer | Product | Quantity | Amount | Status`

### 4. Activate the server and copy the endpoint
1. Open the **MCP Server Trigger** node and note the **Production URL**, e.g.
   `https://<your-n8n-host>/mcp/<path-id>`
2. Authentication is set to **n8n OAuth2**.
3. **Activate** the workflow (the production URL only works while it is active).

### 5. Configure the client
In the **Basic MCP Server Client Multi Tool** workflow, open the **MCP Client** node:

- **Endpoint URL**: the production URL from step 4
- **Server Transport**: `SSE` (as configured; switch to *HTTP Streamable* if your n8n version and server support it)
- **Authentication**: MCP OAuth2 API credential
- **Tools to include**: *Selected* → all 6 tools

### 6. Update the agent's system prompt (recommended)
The current system message in the **Perception** node describes only the email use case, so the agent may under-use the other five tools. A suggested replacement:

```text
You are a productivity assistant with access to these tools:
- Gmail: send emails. Never invent recipient addresses; ask if missing.
- Google Calendar: create events. Use ISO 8601 date-times with the Asia/Kolkata (+05:30) offset.
  Today's date/time is {{ $now.setZone('Asia/Kolkata').toISO() }}.
- Google Sheets: log sales records (Date, Customer, Product, Quantity, Amount, Status).
- Google Docs: create a new document with a meaningful title.
- SerpApi Google Search: look up current information, news, products or facts.
- Calculator: any arithmetic.

Use a tool only when the request needs it; answer general questions directly.
If required details are missing, ask a short follow-up question before calling a tool.
After a tool call, confirm what was done in one or two sentences.
```

### 7. Activate the client and test
Activate the client workflow and open the chat (**Open chat** button on the chat trigger).

---

## Example prompts

| Prompt | Expected tool |
|---|---|
| "Send an email to alex@example.com saying the Control-M upgrade is complete." | Gmail |
| "Schedule a review meeting tomorrow 3–4 PM titled 'Batch SLA review'." | Google Calendar |
| "Log a sale: today, customer Acme, product Widget, qty 5, amount 2500, status Paid." | Google Sheets |
| "Create a document called 'Q4 Automation Roadmap'." | Google Docs |
| "What's the latest n8n release?" | SerpApi |
| "What is 18% GST on 42,750?" | Calculator |
| "Search for today's INR–USD rate and email it to alex@example.com." | SerpApi → Gmail (multi-step) |

---

## Project structure

```
.
├── README.md
├── REQUIREMENTS.md
└── workflows/
    ├── MCP Server Trigger.json
    └── Basic MCP Server Client Multi Tool.json
```

---

## Known limitations

- **Google Docs creates an empty document** — the node only sets the title. To write content, add a Google Docs *Update* operation (or a sub-workflow tool that creates then inserts text).
- **Calendar times depend on the LLM** — `Start`/`End` are free-text strings; without the current date and timezone in the system prompt, relative times like "tomorrow" may resolve incorrectly.
- **Sheets values are strings** — `Quantity` and `Amount` are typed as `string`, so the sheet stores text unless type conversion is enabled.
- **Unmapped sheet columns** — the schema includes `Test purpose` and a long instruction-text column left over from testing; they are not filled by the agent and can be removed from the sheet.
- **Window Buffer Memory is in-process** — chat history is lost on restart and is not shared across instances. Use Postgres/Redis chat memory for persistence.

---

## Security notes

- Keep the MCP Server Trigger behind **OAuth2** (already enabled). Anyone with access to the endpoint can send email from your Gmail account.
- Gmail is an outbound action: consider adding a human-in-the-loop approval step (e.g. n8n *Send and Wait for Approval*) before sending.
- Do not commit exported workflows with real folder IDs, sheet IDs, calendar emails or endpoint URLs to a public repository — replace them with placeholders.
- Scope Google OAuth to the minimum APIs needed and rotate OpenAI/SerpApi keys periodically.
- Set a spending limit on the OpenAI account and a usage cap on SerpApi.

---

## Possible extensions

- Add Control-M Automation API tools (job status, rerun, order) as additional MCP tools for an ops assistant.
- Add a *Read row / lookup* Sheets tool so the agent can answer questions about existing sales data.
- Add Slack or Microsoft Teams notification tools.
- Connect the same MCP endpoint to Claude Desktop or another MCP client to reuse the tools outside n8n.

---

## Author

Deno Melvin Paul D — GenAI Architect program project (n8n + MCP + agentic AI).
