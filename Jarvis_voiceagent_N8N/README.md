# Jarvis – Voice AI Agent on Telegram (n8n + OpenAI)

A Jarvis-style personal assistant on Telegram, built in n8n. You can **talk to it** with a voice note or **type** a message. It understands you, uses tools when needed (web search, email, calculator), and replies twice: **as a text message and as a spoken audio reply**.

## What it can do

- 🎙️ **Understand voice notes:** transcribes them with OpenAI speech-to-text.
- 💬 **Understand text messages.**
- 🔎 **Search the web** for current information, news and prices (SerpAPI).
- ✉️ **Send emails** through Gmail.
- 🧮 **Do calculations** (totals, EMIs, percentages) with a calculator tool, not by guessing.
- 🧠 **Remember the conversation** for follow-up questions.
- 🔊 **Talk back:** turns the answer into a short spoken summary with OpenAI text-to-speech.
- 👋 **Knows your name:** your Telegram first name is passed to the agent.

## How it works

```
                                  ┌─► Get a file ─► STT ─► audio_var ─┐
Telegram Trigger ─► Switch ──Voice┘   (download)  (Whisper) (text)    │
                           └─Text───► Input_var ──────────────────────┤
                                       (text)                         ▼
                                                                  AI Agent
                                            ┌──────────┬───────────┼──────────┬─────────┐
                                     OpenAI Chat   Simple       SerpAPI     Gmail    Calculator
                                     gpt-4o-mini   Memory       (search)    (send)
                                                                   │
                                     ┌─────────────────────────────┴───────────┐
                                     ▼                                         ▼
                                  Telegram                              Basic LLM Chain
                               (text reply)                         (rewrite for speech,
                                                                     OpenAI Chat Model1)
                                                                               │
                                                                       TTS (OpenAI audio)
                                                                               │
                                                                     Telegram1 (send audio)
```

### Step by step

| # | Node | Type | What it does |
|---|------|------|--------------|
| 1 | **Telegram Trigger** | Telegram Trigger | Fires on every message sent to the bot. |
| 2 | **Switch** | Switch | Routes **Voice** messages (has `voice.file_id`) one way and **Text** messages the other. |
| 3a | **Get a file** | Telegram (File → Get) | Voice path: downloads the voice note from Telegram. |
| 4a | **STT** | OpenAI (Audio → Transcribe) | Voice path: converts speech to text. |
| 5a | **audio_var** | Edit Fields (Set) | Voice path: stores the transcript in a `text` field. |
| 3b | **Input_var** | Edit Fields (Set) | Text path: copies the message into the same `text` field. |
| 6 | **AI Agent** | AI Agent | Gets `text` and the user's first name, follows the system prompt, and calls tools when needed. |
| 7 | **OpenAI Chat Model** | OpenAI Chat Model (`gpt-4o-mini`) | The agent's brain. |
| 8 | **Simple Memory** | Simple Memory (window buffer) | Keeps recent conversation history. |
| 9 | **SerpAPI** | SerpAPI Tool (built in) | Google search for current information. |
| 10 | **Gmail** | Gmail Tool | Sends email. The agent fills **To**, **Subject** and **Message**. |
| 11 | **Calculator** | Calculator Tool | Evaluates maths expressions. |
| 12 | **Telegram** | Telegram (Send Message) | Sends the full answer as text. |
| 13 | **Basic LLM Chain** + **OpenAI Chat Model1** | LLM Chain (`gpt-4o-mini`) | Rewrites the answer into a short, natural summary that sounds good spoken. |
| 14 | **TTS** | OpenAI (Audio → Generate) | Turns the summary into an audio file. |
| 15 | **Telegram1** | Telegram (Send Audio) | Sends the audio reply to the same chat. |

Both input paths end with the same `text` field, so the AI Agent handles voice and text the same way.

### Agent behaviour (system prompt)

| Tool | Used when… |
|------|------------|
| **SerpAPI** | The user asks for the latest information, current data, recent news, or anything that needs the internet. |
| **Gmail** | The user wants to send an email. Uses the format *Dear {name}, {body}, Regards, {sender or "Team"}* and replies with a short status ("Email sent" / "Email failed"). |
| **Calculator** | The user asks for any calculation, simple or complex. |
| No tool | General conversation and knowledge questions. |

General rules: call a tool only when needed, never pretend a tool can do more than it can, and keep tool calls precise.

### Example conversation

```
You (voice): "Hey, what's 18% GST on 42,500 rupees?"

Jarvis (text):  18% GST on ₹42,500 is ₹7,650, making the total ₹50,150.
Jarvis (audio): 🔊 "Eighteen percent GST on forty-two thousand five hundred
                 rupees is seven thousand six hundred fifty, so the total
                 comes to fifty thousand one hundred fifty."

You (text): Email that to accounts@example.com

Jarvis:     Email sent.  + 🔊 audio confirmation
```

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. In short:

- An n8n instance reachable over **public HTTPS** (Telegram needs this)
- A Telegram bot token from @BotFather
- An OpenAI API key with billing enabled (used for chat, speech-to-text and text-to-speech)
- A Google account with the Gmail API enabled
- A SerpAPI API key

**No community nodes are needed.** This workflow uses n8n's built-in SerpAPI tool.

## Setup

### 1. Import the workflow

1. Create a new workflow in n8n.
2. Use **⋯ menu → Import from File** and choose `Voice_AI_Agent.json`.

### 2. Create the credentials

| Credential | Used by |
|------------|---------|
| Telegram API (bot token from @BotFather) | Telegram Trigger, Get a file, Telegram, Telegram1 |
| OpenAI API | OpenAI Chat Model, OpenAI Chat Model1, STT, TTS |
| Gmail OAuth2 | Gmail |
| SerpApi (API key from serpapi.com) | SerpAPI |

**Gmail setup:** in Google Cloud Console, enable the **Gmail API**, configure the OAuth consent screen (add yourself as a test user), create an OAuth client ID (Web application) with n8n's redirect URL, then click **Sign in with Google** in the n8n credential. On n8n Cloud, signing in is usually enough.

### 3. Activate and test

1. Toggle the workflow to **Active**.
2. In Telegram, send your bot a **text** message, then a **voice note**.
3. You should get a text reply followed by an audio reply each time.
4. In n8n, check **Executions** to see which path ran and which tools were called.

## Configuration

| What to change | Where | Default |
|----------------|-------|---------|
| Agent personality and tool rules | AI Agent → Options → System Message | Tool-calling prompt |
| Agent model | OpenAI Chat Model → Model | `gpt-4o-mini` |
| How the spoken summary is written | Basic LLM Chain → Messages | Concise, speech-friendly |
| Voice, speed, TTS model | TTS → Options | OpenAI defaults |
| Transcription language or prompt | STT → Options | Auto-detect |
| Memory length | Simple Memory → Context Window Length | Node default |

To make it feel more like Jarvis, add a persona line to the top of the system message, for example: *"You are Jarvis, a calm, witty personal assistant. Address the user by name."*

## Known limitations and suggested improvements

- **⚠️ Every user shares the same memory (bug).** Simple Memory's session key is `telegram_ {{ $json.message.chat.id }}`. When the AI Agent runs, `$json` is the output of *Input_var* or *audio_var*, which only contain `text`, so the chat ID is empty and every chat uses the key `telegram_ `. Different users will see each other's context. To fix it, change the session key to:
  ```
  telegram_{{ $('Telegram Trigger').item.json.message.chat.id }}
  ```
- **⚠️ The bot is open to anyone.** Anyone who finds it can send email from your Gmail account and spend your OpenAI and SerpAPI credits. Add an **IF** node after the Telegram Trigger that only allows your chat ID.
- **Every reply comes as text and audio,** even for typed messages. If you want audio only for voice input, add an **IF** after the AI Agent that sends to *Basic LLM Chain* only when the original message was a voice note.
- **The audio is a summary, not the full answer.** *Basic LLM Chain* shortens the reply for speech, so details like links appear only in the text message. This is intentional; adjust its prompt if you want the full answer read out.
- **Other message types fall into the Text path.** The Text rule (`message.text || ""` with *exists*) matches any message, so stickers, photos and documents reach the agent with empty text. Change the Text rule to *is not empty*, and add a fallback reply for unsupported types.
- **The email-format rules don't match the tool fields.** The system prompt talks about `to`, `subject`, `body` and `fromName`, but the Gmail tool's fields are **To**, **Subject** and **Message**. It usually works, but aligning the wording (and dropping the `fromName` idea) makes tool calls more reliable.
- **No confirmation before sending email.** Consider asking the agent to read back the draft and wait for "yes".
- **The agent doesn't know today's date.** Add `Current date and time: {{ $now.setZone('Asia/Kolkata').toFormat('dd LLL yyyy, hh:mm a') }}` to the prompt for questions like "what's the date" or "news from this week".
- **Older node versions.** The AI Agent (v2), Telegram Trigger (v1.2) and Gmail tool (v2.1) are older versions. They still work, but newer n8n versions offer updated versions with more options.
- **Memory is temporary.** Simple Memory is lost when n8n restarts. Use Postgres or Redis chat memory for persistence.
- **Telegram limits:** voice files must be under 20 MB to download, and text replies over 4,096 characters will fail.

## Privacy and cost

- Voice notes, messages and replies are sent to OpenAI. Search queries go to SerpAPI.
- Each message can use up to 5 paid calls: transcription (voice only), the agent model, any tool calls, the summary model, and text-to-speech.
- Never commit bot tokens, API keys or OAuth secrets. n8n stores them in its credential store, and the exported JSON only contains credential references.

## Files

| File | Purpose |
|------|---------|
| `Voice_AI_Agent.json` | The n8n workflow export |
| `README.md` | This guide |
| `requirements.txt` | Prerequisites, accounts and permissions |
