# Requirements

## Platform

| Item | Requirement |
|---|---|
| n8n | Self-hosted (tested on a Hostinger VPS) with the AI Agent v3.x node (supports a fallback model) |
| Access | Public **HTTPS** domain. Telegram webhooks require HTTPS. |
| Environment | `WEBHOOK_URL` set to the public HTTPS URL |
| Workflow state | Published (active) so the Telegram webhook is registered |

## n8n nodes used

| Node | Type | Version | Package |
|---|---|---|---|
| Telegram Trigger | `n8n-nodes-base.telegramTrigger` | 1.5 | Built-in |
| Telegram | `n8n-nodes-base.telegram` | 1.2 | Built-in |
| AI Agent | `@n8n/n8n-nodes-langchain.agent` | 3.1 | Built-in |
| AI Agent Tool | `@n8n/n8n-nodes-langchain.agentTool` | 3 | Built-in |
| OpenAI Chat Model (×4) | `@n8n/n8n-nodes-langchain.lmChatOpenAi` | 1.3 | Built-in |
| Simple Memory | `@n8n/n8n-nodes-langchain.memoryBufferWindow` | 1.4 | Built-in |
| SerpApi Tool (×2) | `n8n-nodes-serpapi.serpApiTool` | 1 | **SerpApi node package** – install it if missing (*Settings → Community nodes*, or from the nodes panel) |

## Credentials

| Credential | Type in n8n | Used by | How to get it |
|---|---|---|---|
| Telegram account | Telegram API | Telegram Trigger, Send a text message | Bot token from @BotFather |
| OpenAI account | OpenAI API | All 4 OpenAI Chat Model nodes | API key from platform.openai.com (billing enabled) |
| SerpApi account | SerpApi | Amazon search, Google search | API key from serpapi.com |

## External services

| Service | Used for | Notes |
|---|---|---|
| Telegram Bot API | User interface | One webhook per bot token; message limit 4096 characters |
| OpenAI | `gpt-4.1-mini` for the parent, both specialists, and the fallback | Up to 3 or more LLM calls per message when specialists are used |
| SerpApi – Amazon engine | Product search on `amazon.in` | Each search uses SerpApi quota |
| SerpApi – Google engine | Web research | Each search uses SerpApi quota |

## Configuration checklist

- [ ] Telegram bot created and token added to the `Telegram account` credential
- [ ] SerpApi node package installed and the API key added
- [ ] OpenAI credential attached to all 4 chat model nodes
- [ ] Amazon search tool domain = `amazon.in`
- [ ] Simple Memory session key = `{{ $json.message.chat.id }}`
- [ ] Workflow **Published**
- [ ] Test messages: one product query, one current-news query, one greeting
