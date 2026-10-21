# Requirements

## Platform

| Item | Requirement |
|---|---|
| n8n | Self-hosted (tested on a Hostinger VPS) with **instance-level MCP server** support and the *Available in MCP* workflow setting |
| Access | Public HTTPS domain for the n8n instance (e.g. `https://<your-n8n-domain>`) |
| Environment | `WEBHOOK_URL` set to the public HTTPS URL, so webhook and MCP URLs don't show `localhost` |
| Reverse proxy | If using nginx/Traefik: no response buffering or compression on MCP paths (needed for streaming) |

## n8n nodes used

| Node | Type | Version |
|---|---|---|
| Chat Trigger | `@n8n/n8n-nodes-langchain.chatTrigger` | 1.5 |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 |
| OpenAI Chat Model | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 |
| Window Buffer Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.4 |
| MCP Client Tool | `@n8n/n8n-nodes-langchain.mcpClientTool` | 1.4 |
| Webhook | `n8n-nodes-base.webhook` | 2.2 |
| Manual Trigger | `n8n-nodes-base.manualTrigger` | 1 |
| Gmail | `n8n-nodes-base.gmail` | 2.2 |
| Aggregate | `n8n-nodes-base.aggregate` | 1 |

All of these are built-in n8n nodes. No community nodes are required.

## Credentials

| Credential | Type in n8n | Used by | Notes |
|---|---|---|---|
| OpenAI account | OpenAI API | Brain, OpenAI Chat Model | API key with access to `gpt-4.1-mini` |
| Gmail account | Gmail OAuth2 | Get many messages, Send a message | Needs read and send scopes |
| MCP N8N Instance | MCP OAuth2 | MCP Client N8N Instance | Authorizes the client against the instance's MCP server |

## External services

### OpenAI
- An active API account with billing enabled
- Model: `gpt-4.1-mini` (used by both agents)

### Google Cloud (for Gmail)
- A Google Cloud project with the **Gmail API** enabled
- An OAuth consent screen, plus an OAuth 2.0 Client ID of type **Web application**
- **Authorized redirect URI** = the OAuth callback URL shown in the n8n Gmail credential, which must use the public domain
- Publishing status set to **In production**, to avoid refresh tokens expiring after 7 days. Unverified is fine for personal use.

## Workflow configuration checklist

- [ ] Summarizer workflow has a **Webhook** trigger (POST, path `summarize-unread`)
- [ ] Summarizer workflow: **Available in MCP** = on
- [ ] Summarizer workflow: **Published**
- [ ] Chat workflow: MCP Client endpoint = `https://<your-n8n-domain>/mcp-server/http`
- [ ] Chat workflow: system message includes the Webhook trigger + `{"email": ...}` body instruction
- [ ] Chat workflow: **Published**
- [ ] All three credentials connected and tested
