# Requirements

## Platform

| Item | Requirement |
|---|---|
| n8n | Self-hosted (tested on a Hostinger VPS); a version that includes **human-in-the-loop tool approval** (`telegramHitlTool`) and AI Agent v3.x |
| Access | Public **HTTPS** domain (Telegram webhooks and approval callbacks need it) |
| Environment | `WEBHOOK_URL` set to the public HTTPS URL |
| Workflow state | Published (active) |

## n8n nodes used

| Node | Type | Version |
|---|---|---|
| Telegram Trigger | `n8n-nodes-base.telegramTrigger` | 1.5 |
| Telegram (send) | `n8n-nodes-base.telegram` | 1.2 |
| Telegram HITL Tool | `n8n-nodes-base.telegramHitlTool` | 1.2 |
| Gmail Tool | `n8n-nodes-base.gmailTool` | 2.2 |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 |
| OpenAI Chat Model | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 |
| Simple Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.4 |

All of these are built-in nodes. No community packages are needed.

## Credentials

| Credential | Type | Used by | How to get it |
|---|---|---|---|
| Telegram account | Telegram API | Trigger, HITL tool, Send a text message | Bot token from @BotFather |
| OpenAI account | OpenAI API | Brain | API key from platform.openai.com |
| Gmail account | Gmail OAuth2 | Action | Google Cloud OAuth client (Web application) with the Gmail API enabled |

## Google Cloud (Gmail)

- Gmail API enabled in the project
- OAuth redirect URI = the callback URL shown in the n8n Gmail credential (public domain)
- Consent screen publishing status **In production**, so tokens don't expire every 7 days

## Configuration checklist

- [ ] Memory: Session ID = **Define below**, Key = `{{ $('Telegram Trigger').item.json.message.chat.id }}`
- [ ] Gmail tool (Action) connected **to the Telegram HITL node**, not directly to the AI Agent
- [ ] HITL approval type = double; labels set
- [ ] HITL chat ID = requester chat (default) or a supervisor/group chat ID
- [ ] All three credentials connected
- [ ] Workflow **Published**
- [ ] Test: send → approve, send → cancel, draft only, general question
