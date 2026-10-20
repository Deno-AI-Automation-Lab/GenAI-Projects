# Unread Email Summariser (n8n + OpenAI)

An n8n workflow that checks a Gmail inbox every 6 hours, picks up the latest unread emails, asks an AI agent to summarise each one in a single line, and emails the digest back to you.

## How it works

```
Schedule Trigger ──► Get many messages ──► Aggregate ──► AI Agent ──► Send a message
  (every 6 hrs)      (Gmail, UNREAD, 5)   (into "emails")    │         (Gmail digest)
                                                             │
                                                    OpenAI Chat Model
                                                      (gpt-4.1-mini)
```

| # | Node | Type | What it does |
|---|------|------|--------------|
| 1 | **Schedule Trigger** | Schedule Trigger | Starts the workflow every 6 hours. |
| 2 | **Get many messages** | Gmail (Message → Get Many) | Fetches up to 5 messages with the `UNREAD` label. |
| 3 | **Aggregate** | Aggregate | Combines the separate email items into one item with an `emails` array, so the AI sees all emails in a single prompt. |
| 4 | **AI Agent** | AI Agent (LangChain) | Summarises each email in one line (max 20 words, no greeting, no numbering). |
| 5 | **OpenAI Chat Model** | OpenAI Chat Model | The LLM behind the agent, using `gpt-4.1-mini`. |
| 6 | **Send a message** | Gmail (Message → Send) | Emails the summaries as plain text with the subject *"Latest 5 Unread Email Summaries"*. |

### Sample output email

```
Subject: Latest 5 Unread Email Summaries

Latest 5 Unread Email Summaries

Team meeting moved to Thursday 3 PM; agenda covers Q4 automation roadmap.
Invoice #4821 due on 25 October; payment link included.
Control-M job PAYROLL_EOD failed overnight; rerun requested before 9 AM.
Webinar invite: Agentic AI in enterprise operations, 28 October, free registration.
Password for HR portal expires in 3 days; reset recommended.
```

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. In short:

- An n8n instance (n8n Cloud or self-hosted) recent enough to include the AI Agent node v3.x
- A Google account with Gmail, plus a Google Cloud OAuth client for the Gmail API
- An OpenAI API key with billing enabled

## Setup

### 1. Import the workflow

1. Open n8n and create a new workflow.
2. Use **⋯ menu → Import from File** and choose `Unread_Email_Summariser.json`.
   (Or copy the JSON contents and paste them straight onto the canvas.)

### 2. Create the Gmail credential

1. In Google Cloud Console, create a project and enable the **Gmail API**.
2. Configure the OAuth consent screen and add your Gmail address as a test user.
3. Create an **OAuth client ID** (type: Web application) and add n8n's OAuth redirect URL as an authorised redirect URI. n8n shows this URL in the credential dialog.
4. In n8n, create a **Gmail OAuth2** credential, paste the Client ID and Client Secret, and click **Sign in with Google**.
5. Select this credential in both Gmail nodes: **Get many messages** and **Send a message**.

> On n8n Cloud, you can usually just click **Sign in with Google** without creating your own OAuth client.

### 3. Create the OpenAI credential

1. Generate an API key at platform.openai.com.
2. In n8n, create an **OpenAI** credential with the key.
3. Select it in the **OpenAI Chat Model** node.

### 4. Set the recipient

Open **Send a message** and change **To** to the address that should receive the digest.

### 5. Test, then activate

1. Click **Execute workflow** and check each node's output.
2. Confirm the summary email arrives.
3. Toggle the workflow to **Active** so the schedule runs it.

## Configuration

| What to change | Where | Default |
|----------------|-------|---------|
| Run frequency | Schedule Trigger → Hours Between Triggers | 6 |
| Number of emails | Get many messages → Limit | 5 |
| Which emails | Get many messages → Filters (labels, search query, dates) | `UNREAD` label |
| Summary style | AI Agent → Prompt | 1 line, ≤ 20 words each |
| Model | OpenAI Chat Model → Model | `gpt-4.1-mini` |
| Recipient / subject | Send a message | Your address / "Latest 5 Unread Email Summaries" |

If you change the **Limit**, update the numbers in the AI Agent prompt ("Summarize each of the 5 emails…", "Return exactly 5 lines") and in the email subject so they match.

## Known limitations and suggested improvements

- **The same emails can be summarised again.** The workflow doesn't mark emails as read, so unread mail shows up in every run until you open it. To fix this, add a Gmail **Message → Mark as Read** step after sending, or filter with a search query like `is:unread newer_than:6h`.
- **The prompt assumes exactly 5 emails.** If fewer than 5 are unread, the model may pad or repeat lines. Use a prompt that says "one line per email" without a fixed count.
- **Nothing to summarise still sends an email.** Add an **IF** node after *Get many messages* to stop when no unread emails are found.
- **Only the email snippet is summarised.** *Simplify* is left at its default (on), so the AI gets a short snippet rather than the full body. Turn *Simplify* off for better summaries of long emails; this increases token usage.
- **Sender context.** Including the sender and subject in each line (for example, `Sender – Subject: summary`) makes the digest easier to scan.
- **Error handling.** Set an **Error Workflow** in the workflow settings to get notified if Gmail or OpenAI calls fail.

## Privacy and cost

- Email content is sent to OpenAI for summarisation. Don't use this on mailboxes whose content must not leave your organisation without approval.
- Cost depends on email length and frequency. At 5 emails every 6 hours with `gpt-4.1-mini`, usage is small, but check current OpenAI pricing for your plan.
- Never commit API keys or OAuth secrets. n8n stores them in its credential store, and the exported JSON only contains credential references.

## Files

| File | Purpose |
|------|---------|
| `Unread_Email_Summariser.json` | The n8n workflow export |
| `README.md` | This guide |
| `requirements.txt` | Prerequisites, accounts and permissions |
