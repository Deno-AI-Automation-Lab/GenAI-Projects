# Requirements

## Platform

| Item | Requirement |
|---|---|
| n8n | Self-hosted (tested on a Hostinger VPS) with AI Agent v3.x |
| Access | Public **HTTPS** domain, so the web app can reach the webhook |
| Environment | `WEBHOOK_URL` set to the public HTTPS URL; instance timezone ideally `GENERIC_TIMEZONE=Asia/Kolkata` |
| Workflow state | Published (Production webhook URL active) |
| Frontend | Web app built on **Emergent**, configured with the n8n Production webhook URL |

## n8n nodes used

| Node | Type | Version | Package |
|---|---|---|---|
| Webhook | `n8n-nodes-base.webhook` | 2.2 | Built-in |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 | Built-in |
| OpenAI Chat Model | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 | Built-in |
| Simple Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.4 | Built-in |
| Google Docs Tool | `n8n-nodes-base.googleDocsTool` | 2 | Built-in |
| Google Calendar Tool | `n8n-nodes-base.googleCalendarTool` | 1.3 | Built-in |
| Google Sheets Tool | `n8n-nodes-base.googleSheetsTool` | 4.7 | Built-in |
| Gmail Tool | `n8n-nodes-base.gmailTool` | 2.2 | Built-in |
| SerpApi Tool | `n8n-nodes-serpapi.serpApiTool` | 1 | **SerpApi node package** (install if missing) |

## Credentials

| Credential | Type | Used by |
|---|---|---|
| OpenAI account | OpenAI API | OpenAI Chat Model (`gpt-4.1-mini`) |
| Google Docs account | Google Docs OAuth2 | Create a document |
| Google Calendar account | Google Calendar OAuth2 | Create an event |
| Google Sheets account | Google Sheets OAuth2 | Append row |
| Gmail account | Gmail OAuth2 | Send a message |
| SerpApi account | SerpApi | Google search |

## Google Cloud

- APIs enabled: **Google Docs API, Google Drive API, Google Calendar API, Google Sheets API, Gmail API**
- OAuth 2.0 Client (Web application) with the redirect URI from the n8n credential (public domain)
- Consent screen publishing status **In production**, to avoid 7-day token expiry

## Google resources

| Resource | Needed |
|---|---|
| Drive folder | Target folder for new Docs (folder ID set in the Docs tool) |
| Calendar | Calendar to write events to |
| Spreadsheet | Sheet `SalesData` with headers `Date, Customer, Product, Quantity, Amount, Status` |

## Configuration checklist

- [ ] All 6 credentials connected and tested
- [ ] Docs folder ID, calendar, and spreadsheet/sheet point to your own resources
- [ ] Webhook path `Multiagent`, Respond = When Last Node Finishes
- [ ] CORS Allowed Origins set to the Emergent app's domain (if the browser calls n8n directly)
- [ ] Emergent app uses the **Production** webhook URL and sends `?message=`
- [ ] Workflow **Published**
- [ ] Recommended: Header Auth on the webhook; per-user memory session key
- [ ] Test each tool: Docs, Calendar, Sheets, Gmail, Search, and "what time is it"
