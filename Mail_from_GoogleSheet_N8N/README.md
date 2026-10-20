# 📧 Mail From Google Sheet (n8n Workflow)

An **n8n** automation that reads order rows from a Google Sheet and sends an **"Order Update" email
through Gmail** for each row, either on demand or automatically whenever a new row is added.

## Workflow

```
Manual trigger ──► Get row(s) in sheet ──┐
                                         ├──► Edit Fields (build subject + body) ──► Gmail: Send a message
Google Sheets Trigger (row added) ───────┘
```

| Node | Type | What it does |
|---|---|---|
| When clicking 'Execute workflow' | Manual Trigger | Runs the workflow on demand |
| Get row(s) in sheet | Google Sheets | Reads **all** rows from the `SalesData` tab |
| Google Sheets Trigger | Google Sheets Trigger | Checks **every minute** for newly added rows |
| Edit Fields | Set | Builds `Subject` and `body` from the row's columns |
| Send a message | Gmail | Sends a plain-text email to the configured recipient |

## Google Sheet format

Tab name: **`SalesData`**

| Date | Customer | Product | Quantity | Amount | Status |
|---|---|---|---|---|---|
| 2026-10-20 | Ravi Traders | Laptop | 2 | 85000 | Shipped |

## Email produced

**Subject:** `Order Update - Ravi Traders`

```
Hello,

Here are the order details:

Date: 2026-10-20
Customer: Ravi Traders
Product: Laptop
Quantity: 2
Amount: ₹85000
Status: Shipped

Regards,
GenAI Automation Lab
```

Any blank column shows as `EMPTY`.

## Requirements

- **n8n**, either self-hosted (npm or Docker) or n8n Cloud
- A **Google account** that can access the sheet
- Three OAuth2 credentials in n8n: **Google Sheets**, **Google Sheets Trigger** and **Gmail**

(See `requirements.txt`. There are no Python packages to install.)

## Setup

### 1. Run n8n

```bash
# Docker
docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n

# or npm
npx n8n
```

Open **http://localhost:5678**.

### 2. Import the workflow

**Workflows → Import from File → `Mail_From_GoogleSheet.json`**

### 3. Connect your credentials

The credential IDs in the JSON belong to the original n8n instance, so reconnect them after importing:

1. Open **Get row(s) in sheet** and create or select a **Google Sheets OAuth2** credential.
2. Open **Google Sheets Trigger** and create or select a **Google Sheets Trigger OAuth2** credential.
3. Open **Send a message** and create or select a **Gmail OAuth2** credential.

For self-hosted n8n, create an OAuth client in Google Cloud Console, enable the **Google Sheets**,
**Google Drive** and **Gmail** APIs, and add n8n's OAuth redirect URL. n8n Cloud handles this for you.

### 4. Point it at your sheet and recipient

- In both Google Sheets nodes, pick **your** spreadsheet and the `SalesData` tab.
- In **Send a message**, set **To** to the address that should receive the emails.

### 5. Test, then activate

- Click **Execute workflow**. You get one email per row in the sheet.
- Toggle the workflow **Active**. From then on, each new row triggers an email within about a minute.

## ⚠️ Things to know

| Point | Detail |
|---|---|
| Manual run emails **every** row | "Get row(s) in sheet" has no filter, so a 50-row sheet sends 50 emails. Use a small test sheet, or add a filter |
| Trigger only reacts to **new** rows | Edits to existing rows don't send anything (the event is `rowAdded`) |
| `0` shows as `EMPTY` | The expressions use `||`, so a Quantity or Amount of `0` becomes `EMPTY`. Use `??` instead if zero is a valid value |
| Polling every minute | Fine for testing; for production, a longer interval reduces API calls |
| Gmail sending limits | Consumer Gmail accounts have daily send limits, so batch large sheets carefully |
| Fixed recipient | Every email goes to one address. To email each customer, add an `Email` column and set **To** to `{{ $json.Email }}` |

## Ideas for next steps

- Add an **If** node to email only rows where `Status` is, for example, `Shipped`
- Write a `Sent` timestamp back to the sheet to avoid duplicate emails
- Switch to an HTML email template
- Add an AI node to write a personalised message per customer

## Project layout

```
.
├── Mail_From_GoogleSheet.json   # n8n workflow export
├── requirements.txt             # prerequisites (no Python packages)
└── README.md
```
