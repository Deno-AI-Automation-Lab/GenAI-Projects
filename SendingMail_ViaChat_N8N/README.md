# Sending Mail – Conversational Email Agent (n8n + OpenAI + Gmail)

A chat-based AI agent built in n8n. You type a request in plain English, such as *"Email ravi@example.com that tomorrow's deployment is moved to 7 PM"*, and the agent works out the recipient, subject and message, then sends the email through Gmail. If you just ask a general question, it answers in the chat without sending anything.

## How it works

```
When chat message received ──► Perception (AI Agent)
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
         Brain (LLM)          Memory (buffer)      Action (Gmail tool)
        gpt-4.1-mini        remembers recent       sends the email
                             chat messages
```

The nodes are named after the parts of the classic AI agent architecture:

| Node name | n8n node type | Agent role | What it does |
|-----------|---------------|------------|--------------|
| **When chat message received** | Chat Trigger | Input | Opens a chat window and starts the workflow on each message. |
| **Perception** | AI Agent | Perception + reasoning loop | Reads the message, follows the system prompt, decides whether to reply in chat or call the Gmail tool. |
| **Brain** | OpenAI Chat Model (`gpt-4.1-mini`) | Reasoning | The LLM that does the thinking for the agent. |
| **Memory** | Simple Memory (window buffer) | Memory | Keeps the last few messages of the conversation, so follow-ups like *"change the subject to Urgent and send it"* work. |
| **Action** | Gmail Tool (Message → Send) | Action | Sends the email. The agent fills **To**, **Subject** and **Message** itself using `$fromAI()`. |

### Agent behaviour (system prompt)

- Uses the Gmail tool only when you ask to send, compose or email a message.
- Pulls the recipient address, subject and message out of your request.
- Writes a short, relevant subject if you don't give one.
- Answers general questions in chat without using Gmail.
- Never makes up a recipient address.

### Example conversation

```
You:   Send an email to priya@example.com saying the Control-M upgrade
       is scheduled for Saturday 10 PM IST and asking her to confirm
       downtime approval.

Agent: Done. I've sent an email to priya@example.com with the subject
       "Control-M Upgrade – Saturday 10 PM IST" asking her to confirm
       downtime approval.

You:   What's the difference between a cyclic and a regular job?

Agent: (answers in chat, no email sent)
```

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. In short:

- An n8n instance (n8n Cloud or self-hosted) recent enough to include the AI Agent node v3.x and the Chat Trigger
- A Google account with Gmail, plus a Google Cloud OAuth client for the Gmail API
- An OpenAI API key with billing enabled

## Setup

### 1. Import the workflow

1. Open n8n and create a new workflow.
2. Use **⋯ menu → Import from File** and choose `Sending_Mail.json`.
   (Or copy the JSON contents and paste them straight onto the canvas.)

### 2. Create the Gmail credential

1. In Google Cloud Console, create a project and enable the **Gmail API**.
2. Configure the OAuth consent screen and add your Gmail address as a test user.
3. Create an **OAuth client ID** (type: Web application) and add n8n's OAuth redirect URL as an authorised redirect URI. n8n shows this URL in the credential dialog.
4. In n8n, create a **Gmail OAuth2** credential, paste the Client ID and Client Secret, and click **Sign in with Google**.
5. Select this credential in the **Action** node.

> On n8n Cloud, you can usually just click **Sign in with Google** without creating your own OAuth client.

### 3. Create the OpenAI credential

1. Generate an API key at platform.openai.com.
2. In n8n, create an **OpenAI** credential with the key.
3. Select it in the **Brain** node.

## Running it

### In the editor (testing)

1. Open the workflow.
2. Click **Open chat** at the bottom of the canvas.
3. Type a request. For your first test, send an email to your own address.
4. Check the **Logs** panel to see the agent's reasoning and the tool call it made.

### As a shared chat (optional)

1. Open the **When chat message received** node and turn on **Make Chat Publicly Available**.
2. Toggle the workflow to **Active**.
3. Share the chat URL shown in the node.

> ⚠️ Anyone with a public chat URL can send email from your Gmail account. Turn on authentication in the Chat Trigger node (for example, Basic Auth or n8n user auth) before sharing it.

## Configuration

| What to change | Where |
|----------------|-------|
| Agent behaviour and rules | Perception → Options → System Message |
| Model | Brain → Model (default `gpt-4.1-mini`) |
| How much chat history is remembered | Memory → Context Window Length |
| Email format (HTML or plain text) | Action → Email Type |
| Tool instructions the LLM sees | Action → Description |

## Known limitations and suggested improvements

- **Emails are sent without confirmation.** The agent sends as soon as it decides to. To add a safety step, update the system prompt so it shows you a draft and waits for "yes" before calling the tool, or put a human-approval step in front of the Gmail tool if your n8n version supports it.
- **Typo in the tool description.** The Action node's description starts with *"Send a message in GmailUse this tool…"*. Add a period and space after "Gmail" so the LLM reads clean instructions.
- **Email Type isn't set.** The Gmail tool uses its default format. If line breaks look wrong in received emails, set **Email Type** to **Text**, or tell the agent in the system prompt to write HTML.
- **Memory is short and temporary.** Simple Memory keeps only a few recent messages and lives in n8n's memory, so history is lost on restart. For longer or persistent memory, switch to Postgres or Redis chat memory.
- **Send only.** The agent can't read, reply to, or search emails. Add more Gmail tool nodes (Get Many, Reply, Search) to extend it.
- **No CC/BCC or attachments.** Add `$fromAI()` fields for CC and BCC in the Action node if you need them.

## Privacy and security

- Your chat messages and email content are sent to OpenAI to be processed.
- The agent can send mail as your Gmail account. Test with your own address first, and protect the chat if it's public.
- Never commit API keys or OAuth secrets. n8n stores them in its credential store, and the exported JSON only contains credential references.

## Files

| File | Purpose |
|------|---------|
| `Sending_Mail.json` | The n8n workflow export |
| `README.md` | This guide |
| `requirements.txt` | Prerequisites, accounts and permissions |
