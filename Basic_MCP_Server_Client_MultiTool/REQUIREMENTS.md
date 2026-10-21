# Requirements – n8n MCP Server + AI Agent Client

## 1. Platform

| Component | Requirement | Notes |
|---|---|---|
| n8n | Recent 1.x or later, **self-hosted** | Must include `@n8n/n8n-nodes-langchain.mcpTrigger` (v2.1), `mcpClientTool` (v1.4), `agent` (v3.1), `chatTrigger` (v1.5) |
| Hosting | Public **HTTPS** endpoint with valid TLS certificate | Required for Google OAuth callbacks and remote MCP clients (current setup: Hostinger VPS) |
| Runtime | Node.js version supported by your n8n release, or the official n8n Docker image | Docker is recommended |
| Database | SQLite (default) or PostgreSQL | PostgreSQL recommended for production |
| Reverse proxy | Nginx / Traefik / Caddy | Must allow **SSE / long-lived connections** (disable response buffering, raise read timeout) for the MCP endpoint |

### Recommended environment variables

```env
N8N_HOST=<your-n8n-host>
N8N_PROTOCOL=https
WEBHOOK_URL=https://<your-n8n-host>/
GENERIC_TIMEZONE=Asia/Kolkata
TZ=Asia/Kolkata
N8N_ENCRYPTION_KEY=<long-random-secret>
N8N_COMMUNITY_PACKAGES_ENABLED=true
N8N_COMMUNITY_PACKAGES_ALLOW_TOOL_USAGE=true
```

`N8N_COMMUNITY_PACKAGES_ALLOW_TOOL_USAGE` lets the SerpApi community node be used as an AI tool.

## 2. n8n nodes

### Built-in nodes

| Node | Type | Version |
|---|---|---|
| MCP Server Trigger | `@n8n/n8n-nodes-langchain.mcpTrigger` | 2.1 |
| MCP Client Tool | `@n8n/n8n-nodes-langchain.mcpClientTool` | 1.4 |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 |
| Chat Trigger | `@n8n/n8n-nodes-langchain.chatTrigger` | 1.5 |
| OpenAI Chat Model | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 |
| Window Buffer Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.4 |
| Calculator Tool | `@n8n/n8n-nodes-langchain.toolCalculator` | 1 |
| Google Docs Tool | `n8n-nodes-base.googleDocsTool` | 2 |
| Google Calendar Tool | `n8n-nodes-base.googleCalendarTool` | 1.3 |
| Google Sheets Tool | `n8n-nodes-base.googleSheetsTool` | 4.7 |
| Gmail Tool | `n8n-nodes-base.gmailTool` | 2.2 |

### Community node

| Package | Node | Install |
|---|---|---|
| `n8n-nodes-serpapi` | `serpApiTool` | **Settings → Community Nodes → Install** → `n8n-nodes-serpapi` |

## 3. External accounts and API keys

| Service | What you need | Used by |
|---|---|---|
| OpenAI | API key with access to `gpt-4.1-mini`, billing enabled | Brain (LLM) |
| SerpApi | API key (free tier available, limited searches/month) | Google search tool |
| Google Cloud | Project with an OAuth 2.0 Client ID (Web application) | All Google tools |
| Google account | Gmail, Calendar, Drive/Docs and Sheets access | All Google tools |

### Google Cloud APIs to enable

- Gmail API
- Google Calendar API
- Google Sheets API
- Google Docs API
- Google Drive API (needed to create the doc inside a specific folder)

### OAuth consent screen

- Add your Google account as a **test user** while the app is in *Testing* mode
- Authorised redirect URI: `https://<your-n8n-host>/rest/oauth2-credential/callback`

## 4. n8n credentials

| Credential | Type in n8n |
|---|---|
| Google Docs account | Google Docs OAuth2 API |
| Google Calendar account | Google Calendar OAuth2 API |
| Google Sheets account | Google Sheets OAuth2 API |
| Gmail account | Gmail OAuth2 |
| SerpApi account | SerpApi API |
| OpenAI account | OpenAI API |
| MCP client credential | MCP OAuth2 API |

## 5. Google resources to prepare

| Resource | Requirement |
|---|---|
| Drive folder | A folder for generated documents; copy its ID into the Google Docs node `folderId` |
| Calendar | Calendar ID (usually your Gmail address) for the Calendar node |
| Spreadsheet | A spreadsheet with a sheet (e.g. `SalesData`) whose first row contains: `Date`, `Customer`, `Product`, `Quantity`, `Amount`, `Status` |

## 6. Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | The user can interact with the agent through the n8n chat interface. |
| FR-02 | The agent answers general questions directly without calling tools. |
| FR-03 | The agent sends an email via Gmail when asked, using only recipient addresses the user supplied. |
| FR-04 | The agent creates Google Calendar events with title, start, end and optional description. |
| FR-05 | The agent appends sales records (Date, Customer, Product, Quantity, Amount, Status) to Google Sheets. |
| FR-06 | The agent creates a Google Doc with a title derived from the request. |
| FR-07 | The agent performs Google web searches through SerpApi for current information. |
| FR-08 | The agent performs arithmetic using the Calculator tool. |
| FR-09 | The agent can chain multiple tools in one request (e.g. search then email). |
| FR-10 | The agent remembers recent conversation turns within a session. |
| FR-11 | All tools are exposed through a single MCP endpoint usable by other MCP clients. |

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | The MCP endpoint requires OAuth2 authentication. |
| NFR-02 | All traffic is served over HTTPS. |
| NFR-03 | Secrets are stored only in n8n credentials, never in workflow JSON or source control. |
| NFR-04 | The MCP Server workflow stays active for the endpoint to be reachable. |
| NFR-05 | Calendar and date handling uses the Asia/Kolkata timezone. |
| NFR-06 | API usage limits/budgets are set for OpenAI and SerpApi. |
| NFR-07 | Workflow JSON exported for sharing has IDs, emails and URLs replaced with placeholders. |

## 8. Optional / production hardening

- PostgreSQL or Redis chat memory instead of Window Buffer Memory
- Human approval step before sending emails
- n8n Error Trigger workflow to alert on failed executions
- Regular backups of the n8n database and `N8N_ENCRYPTION_KEY`
