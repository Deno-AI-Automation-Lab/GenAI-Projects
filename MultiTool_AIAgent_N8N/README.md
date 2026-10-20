# MultiTool AI Agent – Telegram Assistant for Google Workspace & Web Search (n8n + OpenAI)

A Telegram bot built in n8n that acts as a personal assistant. You send it a request in plain English, and one AI agent decides which tools to use. It can:

- 📄 **Create a Google Doc**
- 📅 **Schedule a Google Calendar event**
- 📊 **Log a sale to a Google Sheet**
- ✉️ **Send an email through Gmail**
- 🔎 **Search Google for current information** (via SerpApi)

…or just chat. It can also chain tools in one request, for example *"Search for the latest Control-M release and email a summary to ravi@example.com"*.

## How it works

```
Telegram Trigger ──► Multitool (AI Agent) ──► Send a text message
  (user message)           │                    (reply to same chat)
                           │
    ┌──────────┬───────────┼────────────┬─────────────┬─────────────┬──────────────┐
    │          │           │            │             │             │              │
 OpenAI     Simple     Google Docs   Google       Google        Gmail         SerpApi
 Chat Model Memory     (create doc)  Calendar     Sheets        (send email)  (Google search)
 gpt-4.1-   (per chat)               (create      (append row
 mini                                 event)       to SalesData)
```

| # | Node | Type | What it does |
|---|------|------|--------------|
| 1 | **Telegram Trigger** | Telegram Trigger | Fires when someone messages the bot. |
| 2 | **Multitool** | AI Agent | Receives the message plus the current India date and time, follows the system prompt, picks one or more tools, and writes the reply. |
| 3 | **OpenAI Chat Model** | OpenAI Chat Model (`gpt-4.1-mini`) | The LLM that reasons and fills in tool parameters. |
| 4 | **Simple Memory** | Simple Memory (window buffer) | Remembers recent messages per Telegram chat ID, so follow-ups like *"make it 4 PM instead"* work. |
| 5 | **Create a document in Google Docs** | Google Docs Tool | Creates a new doc in a fixed Drive folder. The agent chooses the **title**. |
| 6 | **Create an event in Google Calendar** | Google Calendar Tool | Creates an event. The agent fills **Summary**, **Start**, **End** and **Description**. |
| 7 | **Append row in sheet in Google Sheets** | Google Sheets Tool | Adds a row to the **SalesData** sheet with **Date, Customer, Product, Quantity, Amount, Status**. |
| 8 | **Send a message in Gmail** | Gmail Tool | Sends an email. The agent fills **To**, **Subject** and **Message**. |
| 9 | **Google search in SerpApi** | SerpApi Tool (Google engine) | Searches Google. The agent writes the query and an optional location. |
| 10 | **Send a text message** | Telegram (Send Message) | Sends the agent's final answer back to the chat. |

All tool fields marked `$fromAI()` are filled by the LLM from your message.

### What the agent receives

Every message is sent to the agent in this format:

```
User request:
<the Telegram message text>

Current India date and time:
Tuesday, 20 October 2026, 09:06:12 PM
```

The date and time come from n8n at the moment the message arrives (`$now` in the `Asia/Kolkata` time zone). They're also included in the system message. This lets the agent work out relative dates like *"tomorrow"* or *"next Monday"* correctly and answer *"what's the time?"* accurately.

### Agent behaviour (system prompt)

**Tool selection**

| Request contains… | Tool used |
|-------------------|-----------|
| create, write, save, prepare a **document** | Google Docs |
| schedule, book, add, create an **event / meeting** | Google Calendar |
| record, save, add, log **sales** | Google Sheets |
| send, compose, **email** a message | Gmail |
| current or recent info, research, news, websites, products, facts that may have changed | Google Search |
| greeting or normal conversation | No tool, answers directly |

**Date and time rules**

- Uses only the supplied India date and time. It never guesses, recalls it from model knowledge, or takes it from earlier messages in memory.
- Works in `Asia/Kolkata` (IST, UTC+05:30) and resolves relative dates against the supplied time.

**Tool-specific rules**

- **Calendar:** extracts title, start, end and description; asks if an essential date or time is missing.
- **Sheets:** extracts Date, Customer, Product, Quantity, Amount, Status; never invents missing business data.
- **Gmail:** writes a short subject if none is given; never invents a recipient and asks for it if missing.
- **Search:** prefers a web search over model knowledge for anything current or changing; never invents links, dates, facts or sources.

**Honesty and general rules**

- Never claims a document, event, record or email was created unless the tool actually succeeded.
- Never pretends a tool was used, and never invents tool results.
- Uses several tools only when the task genuinely needs them.
- Follows the user's latest instruction when it conflicts with older context.
- Keeps replies concise and briefly confirms each completed action.

### Example requests

```
Schedule a review meeting tomorrow 3 PM to 4 PM about Control-M upgrade
→ Google Calendar event created

Log a sale: 20 Oct 2026, customer Acme Ltd, product Laptop, qty 2, amount 120000, status Paid
→ New row in SalesData

Create a document called "Q4 Automation Roadmap"
→ New Google Doc in the configured folder

What's the latest news on n8n AI Agent features?
→ Google search, summarised reply

Find the top 3 Chennai AI training institutes and email the list to priya@example.com
→ Google search + Gmail
```

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. In short:

- An n8n instance reachable over **public HTTPS** (Telegram needs this)
- The **SerpApi community node** (`n8n-nodes-serpapi`) installed
- A Telegram bot token from @BotFather
- A Google account, with the Gmail, Calendar, Sheets, Docs and Drive APIs enabled in Google Cloud
- A SerpApi API key
- An OpenAI API key with billing enabled

## Setup

### 1. Install the SerpApi node

1. In n8n, go to **Settings → Community Nodes → Install**.
2. Enter `n8n-nodes-serpapi` and install it.

Do this before importing, otherwise **Google search in SerpApi** shows as an unknown node.

### 2. Import the workflow

1. Create a new workflow in n8n.
2. Use **⋯ menu → Import from File** and choose `MultiTool_AI_Agent.json`.

### 3. Set up Google Cloud (one-time)

1. In Google Cloud Console, create a project.
2. Enable these APIs: **Gmail API**, **Google Calendar API**, **Google Sheets API**, **Google Docs API**, **Google Drive API**.
3. Configure the OAuth consent screen and add your Google address as a test user.
4. Create an **OAuth client ID** (Web application) and add n8n's OAuth redirect URL as an authorised redirect URI.

> On n8n Cloud, you can usually just click **Sign in with Google** in each credential without creating your own OAuth client.

### 4. Create the credentials in n8n

| Credential | Used by |
|------------|---------|
| Telegram API (bot token from @BotFather) | Telegram Trigger, Send a text message |
| OpenAI API | OpenAI Chat Model |
| Google Docs OAuth2 | Create a document in Google Docs |
| Google Calendar OAuth2 | Create an event in Google Calendar |
| Google Sheets OAuth2 | Append row in sheet in Google Sheets |
| Gmail OAuth2 | Send a message in Gmail |
| SerpApi (API key) | Google search in SerpApi |

Open each node and select the matching credential.

### 5. Point the tools at your own Google resources

The imported workflow still points at the original author's folder, calendar and sheet. Update these:

- **Create a document in Google Docs → Folder ID:** the ID of a Drive folder you own (from the folder URL after `/folders/`).
- **Create an event in Google Calendar → Calendar:** pick your calendar from the list.
- **Append row in sheet in Google Sheets → Document and Sheet:** pick your spreadsheet and tab.

The sheet needs a header row with exactly these column names:

| Date | Customer | Product | Quantity | Amount | Status |
|------|----------|---------|----------|--------|--------|

After choosing your sheet, refresh the column list in the node so the mapping matches your headers.

### 6. Activate and test

1. Toggle the workflow to **Active**.
2. Message your bot in Telegram. Start with a greeting, then try one tool at a time.
3. In n8n, check **Executions** to see which tools the agent called and with what values.

## Configuration

| What to change | Where |
|----------------|-------|
| Tool-selection rules and tone | Multitool → Options → System Message |
| Time zone and date format given to the agent | Multitool → Prompt (User Message) and System Message, the `$now.setZone('Asia/Kolkata')` expression |
| Model | OpenAI Chat Model → Model (default `gpt-4.1-mini`) |
| Conversation memory length | Simple Memory → Context Window Length |
| Docs folder | Create a document in Google Docs → Folder ID |
| Calendar | Create an event in Google Calendar → Calendar |
| Sales sheet and columns | Append row in sheet in Google Sheets |
| What each tool is told to do | Each tool node → Description |

## Known limitations and suggested improvements

- **⚠️ The bot is open to anyone.** Anyone who finds your bot can send email from your Gmail account, add events to your calendar, write to your sheet, and spend your API credits. Add an **IF** node right after the Telegram Trigger that only allows your Telegram chat ID. This is the most important fix before real use.
- **Google Docs are created empty.** The Docs tool only sets the title, so *"write a document about X"* gives a blank doc with the right name. To add content, connect a second Google Docs tool with the **Update** operation (insert text), and have the agent write the body.
- **Calendar events may still land at the wrong hour.** The agent now knows the correct IST date, but the Calendar tool's Start and End are plain text with no time zone attached. If the agent writes `2026-10-21T15:00:00` without `+05:30`, n8n reads it in the workflow's time zone, which may not be IST. Set **Workflow Settings → Timezone** to `Asia/Kolkata`, and add a line to the system message such as *"Always give calendar start and end times in ISO 8601 with the +05:30 offset."*
- **Extra columns in the sheet mapping.** The imported schema includes two leftover columns ("Test purpose" and a long note column) from a test sheet. They stay empty but clutter the mapping. Refresh the columns after selecting your own sheet.
- **Emails and events are created without confirmation.** The agent asks when details are missing, but once it has them it acts straight away. Consider telling it to show a draft and wait for "yes" before sending email or creating events.
- **The date and time are written in two places.** They're in both the user prompt and the system message. If you change the time zone or format, update both.
- **Search location may be blank.** The agent fills SerpApi's *Location* itself. If results look off, set a fixed location (for example, `Chennai, Tamil Nadu, India`) instead of `$fromAI()`.
- **Non-text messages** (stickers, photos, voice notes) have no text, so the agent gets an empty prompt. Filter them with an **IF** node.
- **Telegram formatting.** The agent may reply with `**bold**` markdown, which Telegram shows as raw asterisks because no Parse Mode is set. Ask the agent for plain text, or set Parse Mode.
- **Telegram's 4,096-character limit** can cut off long search summaries. Keep "concise" in the prompt or add a length limit.
- **Memory is temporary.** Simple Memory is lost when n8n restarts. Use Postgres or Redis chat memory for persistence.

## Privacy and cost

- Messages go to OpenAI, search queries go to SerpApi, and actions run on your Google account.
- Each message costs OpenAI tokens; searches also use your SerpApi quota.
- Never commit bot tokens, API keys or OAuth secrets. n8n stores them in its credential store, and the exported JSON only contains credential references. The JSON does include your folder, calendar and sheet IDs, so review it before sharing publicly.

## Files

| File | Purpose |
|------|---------|
| `MultiTool_AI_Agent.json` | The n8n workflow export |
| `README.md` | This guide |
| `requirements.txt` | Prerequisites, accounts and permissions |
