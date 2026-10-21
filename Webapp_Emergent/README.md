# WebApp – Multi-Tool AI Assistant (n8n + Emergent)

A web chat app built with **Emergent** and backed by an **n8n AI agent**. The user types a request in the web app; the app calls an n8n webhook; the **Multitool** agent picks the right tool, performs the action, and returns the answer to the app.

| Tool | What it does |
|---|---|
| Google Docs | Creates a document (title derived from the request) in a fixed Drive folder |
| Google Calendar | Creates events/meetings, resolving relative dates ("tomorrow 4 pm") in IST |
| Google Sheets | Appends sales records (Date, Customer, Product, Quantity, Amount, Status) to the **SalesData** sheet |
| Gmail | Sends emails (asks for the recipient if it's missing) |
| Google Search (SerpApi) | Current information, news, research, websites |

Example requests:

```
Schedule a meeting with the infra team tomorrow 4 pm to 5 pm about patching
Log a sale: customer Ravi, product Laptop, quantity 2, amount 120000, status Paid
Create a document titled "Q4 Automation Plan"
Email someone@example.com that the deployment is complete
What are the latest n8n features?
What time is it now?
```

---

## Architecture

```
┌──────────────────────┐   GET /webhook/Multiagent?message=<text>   ┌───────────────────────────────┐
│  Emergent Web App    │ ─────────────────────────────────────────► │  n8n: Webhook                 │
│  (chat UI)           │                                            │        │                      │
│                      │ ◄───────────────────────────────────────── │        ▼                      │
│  shows `output`      │        JSON: { "output": "<answer>" }      │  Multitool (AI Agent)         │
└──────────────────────┘                                            │   ├─ OpenAI gpt-4.1-mini      │
                                                                    │   ├─ Simple Memory            │
                                                                    │   └─ Tools: Docs · Calendar · │
                                                                    │      Sheets · Gmail · SerpApi │
                                                                    └───────────────────────────────┘
```

## n8n workflow nodes

| Node | Type | Key settings |
|---|---|---|
| Webhook | Webhook v2.2 | Path `Multiagent`, method **GET** (default), Respond: **When Last Node Finishes** |
| Multitool | AI Agent v3.1 | Prompt = `{{ $json.query.message }}` + current IST date/time |
| OpenAI Chat Model | `gpt-4.1-mini` | — |
| Simple Memory | Window Buffer | Session key fixed: `Webapp_1` |
| Create a document in Google Docs | Google Docs Tool | Fixed folder ID; Title via `$fromAI` |
| Create an event in Google Calendar | Google Calendar Tool | Calendar `denomelvin24@gmail.com`; Start, End, Summary and Description via `$fromAI` |
| Append row in sheet in Google Sheets | Google Sheets Tool | Spreadsheet `n8n_google_sheets_email_test`, sheet `SalesData`, 6 mapped columns |
| Send a message in Gmail | Gmail Tool | To, Subject and Message via `$fromAI` |
| Google search in SerpApi | SerpApi Tool | Query + location via `$fromAI` |

### Date and time handling

The current IST date and time is injected into both the prompt and the system message:

```
{{ $now.setZone('Asia/Kolkata').toFormat('EEEE, dd LLLL yyyy, hh:mm:ss a') }}
```

The agent is told to use only this value for "now", to ignore any time stored in memory, and to resolve relative calendar dates from it.

---

## API contract (web app ↔ n8n)

**Request**

```
GET https://<your-n8n-domain>/webhook/Multiagent?message=<URL-encoded user text>
```

**Response** (output of the agent, the last node)

```json
{ "output": "Your meeting 'Patching review' has been scheduled for 22 Oct 2026, 4:00–5:00 PM." }
```

The web app should URL-encode the message and display `output`.

---

## Setup

See [REQUIREMENTS.md](REQUIREMENTS.md) for prerequisites.

### n8n
1. Import `WebApp.json`.
2. Attach the credentials: OpenAI, Google Docs, Google Calendar, Google Sheets, Gmail, SerpApi.
3. Update the resource IDs for your account: the Docs **folder ID**, the **calendar**, and the **spreadsheet / sheet**.
4. Make sure the sheet has the header row: `Date | Customer | Product | Quantity | Amount | Status`.
5. **Publish** the workflow and copy the **Production URL** from the Webhook node.

### Emergent web app
1. Set the n8n Production webhook URL in the app's configuration (e.g. an environment variable such as `N8N_WEBHOOK_URL`).
2. On send: call `GET {N8N_WEBHOOK_URL}?message=<encoded text>` and render `response.output`.
3. Show a loading state, because tool calls can take several seconds.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 404 from webhook | Using the test URL, or the workflow isn't published | Use the **Production URL**; publish |
| Browser CORS error | Webhook doesn't allow the app's origin | Webhook → Options → **Allowed Origins (CORS)** = your app's domain; or call the webhook from the app's backend |
| Empty or undefined prompt | App sends POST or JSON body instead of `?message=` | Send GET with the query param, or change the Webhook to POST and the prompt to `{{ $json.body.message }}` |
| Wrong date in calendar events | Ambiguous request | Include date and time explicitly; the agent uses IST |
| Sheets row lands in the wrong columns | Header row differs from the mapping | Match the headers exactly; re-fetch columns in the node |
| Google tools fail after ~7 days | Google OAuth app in Testing mode | Publish the OAuth app and reconnect the credentials |
| Request times out in the app | Long multi-tool runs | Increase the client timeout; keep requests focused |

## Known limitations and improvements

- **Shared memory**: the session key is fixed (`Webapp_1`), so every user and browser shares one conversation history. Pass a per-user session ID from the app (e.g. `?session=<id>`) and set Key = `{{ $json.query.session }}`.
- **No authentication on the webhook**: anyone with the URL can make the agent send email from your Gmail, create events, or write to your sheet. Add **Header Auth** to the Webhook and send the header from the app's backend, never from browser code.
- **Google Docs creates an empty document** (title only). To add content, follow it with a Docs **Update** tool that inserts text.
- The system message still says *"connected to Telegram"*. Changing it to *"connected to a web app"* avoids confusing the model.
- The SalesData sheet schema contains two extra columns (`Test purpose`, and a long instruction header). They're left empty, which is harmless; remove them if the sheet is used in production.
