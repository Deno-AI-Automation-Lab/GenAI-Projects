# Research Agent – Amazon India Product Research Bot (n8n + Telegram + SerpApi + OpenAI)

A Telegram bot built in n8n that researches products on Amazon India. You message it something like *"Best wireless earbuds under ₹3,000 with ANC"*, and the agent searches Amazon.in live through SerpApi, compares up to 3 products, and replies in Telegram with prices, ratings, key features, links and a recommendation.

## How it works

```
Telegram Trigger ──► Research Agent (AI Agent) ──► Send a text message
 (user message)              │                       (reply to same chat)
          ┌──────────────────┼──────────────────────┐
          │                  │                      │
 OpenAI Chat Model     Simple Memory        Amazon search in SerpApi
   gpt-4.1-mini      (per Telegram chat)      (amazon.in, live results)
```

| # | Node | Type | What it does |
|---|------|------|--------------|
| 1 | **Telegram Trigger** | Telegram Trigger | Fires when someone sends a message to your bot. |
| 2 | **Research Agent** | AI Agent | Takes the message text, follows the system prompt, and decides whether to search Amazon or just reply. |
| 3 | **OpenAI Chat Model** | OpenAI Chat Model (`gpt-4.1-mini`) | The LLM that reasons, writes the search query and the comparison. |
| 4 | **Simple Memory** | Simple Memory (window buffer) | Remembers recent messages, keyed by Telegram chat ID, so each user has their own conversation and follow-ups like *"which of those has better battery?"* work. |
| 5 | **Amazon search in SerpApi** | SerpApi Tool (Amazon engine) | Searches `amazon.in`. The agent writes the search query itself using `$fromAI()`. |
| 6 | **Send a text message** | Telegram (Send Message) | Sends the agent's answer back to the chat the question came from. |

### Agent behaviour (system prompt)

- Must use the Amazon Search tool for any request to find, search, compare, recommend, research or evaluate a product. It doesn't recommend from its own knowledge.
- Compares up to 3 relevant products, with name, price, rating, key features and the Amazon link when the tool returns them.
- Explains which product is best and why.
- Never invents prices, ratings, reviews, features or links.
- Replies to greetings and non-product questions without searching.

### Example conversation

```
You:  Suggest a 1.5 ton 5-star inverter split AC under ₹45,000

Bot:  Here are 3 options from Amazon.in:

      1. <Product name>
         Price: ₹41,990 | Rating: 4.3 ★
         Features: 5-star, copper condenser, convertible 5-in-1 cooling
         Link: https://www.amazon.in/...

      2. <Product name>
         ...

      Best pick: Option 1 – best energy rating and highest rating
      within your budget.
```

*(Product details in real replies come from live SerpApi results.)*

## Prerequisites

See [`requirements.txt`](requirements.txt) for the full list. In short:

- An n8n instance reachable over **public HTTPS** (n8n Cloud, or self-hosted with a domain or tunnel). Telegram triggers need this.
- The **SerpApi community node** (`n8n-nodes-serpapi`) installed in n8n
- A Telegram bot token from @BotFather
- A SerpApi account and API key
- An OpenAI API key with billing enabled

## Setup

### 1. Install the SerpApi node

The **Amazon search in SerpApi** node isn't built into n8n.

1. In n8n, go to **Settings → Community Nodes → Install**.
2. Enter `n8n-nodes-serpapi` and install it.

Install it before importing, otherwise the node shows as unknown.

> On n8n Cloud, it may be available directly from the node panel as a verified node. Search for "SerpApi" when adding a node.

### 2. Import the workflow

1. Create a new workflow in n8n.
2. Use **⋯ menu → Import from File** and choose `Research_Agent.json`.

### 3. Create the Telegram bot and credential

1. In Telegram, open **@BotFather**, send `/newbot`, and follow the prompts.
2. Copy the bot token it gives you.
3. In n8n, create a **Telegram API** credential with that token.
4. Select it in both **Telegram Trigger** and **Send a text message**.

### 4. Create the SerpApi credential

1. Sign up at serpapi.com and copy your API key from the dashboard.
2. In n8n, create a **SerpApi** credential with the key.
3. Select it in **Amazon search in SerpApi**.

### 5. Create the OpenAI credential

1. Generate an API key at platform.openai.com.
2. In n8n, create an **OpenAI** credential with the key.
3. Select it in **OpenAI Chat Model**.

### 6. Activate

1. Toggle the workflow to **Active**. This registers the webhook with Telegram.
2. Open your bot in Telegram, send `/start` or a product question, and wait for the reply.

> To test inside the editor, click **Execute workflow** (n8n then listens for one message), send a message to the bot, and watch each node run. Telegram allows only one webhook per bot, so testing and the active workflow can interrupt each other. If messages stop arriving, deactivate the workflow and activate it again.

## Configuration

| What to change | Where | Default |
|----------------|-------|---------|
| Agent rules and reply format | Research Agent → Options → System Message | Amazon India research rules |
| Amazon marketplace | Amazon search in SerpApi → Amazon Domain | `amazon.in` |
| Model | OpenAI Chat Model → Model | `gpt-4.1-mini` |
| Conversation memory length | Simple Memory → Context Window Length | Node default |
| Number of products compared | System message ("Compare up to 3…") | 3 |

To change the country, set **Amazon Domain** (for example, `amazon.com`) and update "Amazon India" in the system message.

## Known limitations and suggested improvements

- **Anyone who finds the bot can use it**, and each message spends your SerpApi and OpenAI credits. Add an **IF** node after the trigger that only lets your chat ID (or an allow-list) through.
- **Non-text messages can cause errors.** Stickers, photos and voice notes have no `message.text`, so the agent gets an empty prompt. Add an **IF** node that checks text exists and replies "Please send a text message" otherwise.
- **Markdown may show as raw symbols.** The agent may use `**bold**` formatting, but **Send a text message** has no Parse Mode set, so Telegram shows the asterisks. Either tell the agent to reply in plain text, or set Parse Mode to Markdown/HTML. Unescaped special characters can make Telegram reject the message.
- **Telegram's 4,096-character limit.** Long comparisons can fail to send. The "up to 3 products" rule usually keeps replies short enough; add "keep the reply under 3,500 characters" to the prompt for safety.
- **No "typing…" indicator.** Searches take a few seconds. Add a Telegram **Send Chat Action** (typing) node right after the trigger so users know the bot is working.
- **Memory is temporary.** Simple Memory lives in n8n's process and is lost on restart. Use Postgres or Redis chat memory for persistence.
- **Prices and availability change.** Results reflect Amazon at search time, and SerpApi may not return every field (for example, price for out-of-stock items).
- **SerpApi free-plan quota is limited.** Check your monthly search allowance on the SerpApi dashboard.

## Privacy and cost

- User messages go to OpenAI and search queries go to SerpApi.
- Costs per message: one or more SerpApi searches plus OpenAI tokens. Restricting who can use the bot is the best way to control cost.
- Never commit bot tokens or API keys. n8n stores them in its credential store, and the exported JSON only contains credential references.

## Files

| File | Purpose |
|------|---------|
| `Research_Agent.json` | The n8n workflow export |
| `README.md` | This guide |
| `requirements.txt` | Prerequisites, accounts and permissions |
