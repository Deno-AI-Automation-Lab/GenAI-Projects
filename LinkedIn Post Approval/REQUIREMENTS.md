# Requirements

## Platform

| Item | Requirement |
|---|---|
| n8n | Self-hosted (tested on a Hostinger VPS); a version with Gmail **Send and Wait for Response** (custom form) and AI Agent v3.x |
| Access | Public **HTTPS** domain, so the form and the approval links in emails work |
| Environment | `WEBHOOK_URL` = public HTTPS URL; recommended `GENERIC_TIMEZONE=Asia/Kolkata` |
| Execution data | Waiting executions persist in the n8n database (up to 24 h per approval). Use persistent storage, not a throwaway container. |
| Workflow state | **Published** (currently inactive in the export) |

## n8n nodes used

| Node | Type | Version | Count |
|---|---|---|---|
| Form Trigger | `n8n-nodes-base.formTrigger` | 2.6 | 1 |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 | 5 |
| OpenAI Chat Model | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 | 5 |
| Gmail (Send and Wait) | `n8n-nodes-base.gmail` | 2.2 | 4 |
| IF | `n8n-nodes-base.if` | 2.3 | 4 |
| Set (Edit Fields) | `n8n-nodes-base.set` | 3.5 | 2 |
| Google Sheets (Append) | `n8n-nodes-base.googleSheets` | 4.7 | 4 |

All of these are built-in. No community nodes are needed.

## Credentials

| Credential | Type | Used by |
|---|---|---|
| OpenAI account | OpenAI API | 5 OpenAI Chat Model nodes (`gpt-4.1-mini`) |
| Gmail account | Gmail OAuth2 | 4 approval email nodes |
| Google Sheets account | Google Sheets OAuth2 | 4 Append row nodes |

## Google Cloud

- APIs enabled: **Gmail API, Google Sheets API, Google Drive API**
- OAuth 2.0 Client (Web application) with the redirect URI from the n8n credential
- Consent screen **In production**, to avoid 7-day token expiry

## Google resources

| Resource | Needed |
|---|---|
| Spreadsheet | *LinkedIn Post Approval*, sheet *Sheet1*, headers `Date, Topic, Final Post, Status` |
| Approver mailbox | Address in `sendTo` of the 4 Gmail approval nodes |

## Configuration checklist

- [ ] OpenAI, Gmail and Google Sheets credentials connected
- [ ] All 4 Sheets nodes point to your spreadsheet
- [ ] Approver email set in all 4 approval nodes
- [ ] `WEBHOOK_URL` set (approval links must use the public domain)
- [ ] Workflow **Published**; form Production URL bookmarked
- [ ] Test: approve in round 1; reject once then approve; reject 3+ times to exercise the loop
