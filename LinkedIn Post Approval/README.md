# LinkedIn Post Approval – AI Writer with Human Review Loop (n8n)

An n8n workflow that turns a topic into a polished LinkedIn post, using a **Writer → Reviewer** agent pair and a **human approval loop over email**. Each draft is emailed for approval. Rejected drafts are rewritten from your feedback and re-sent until you approve. Approved posts are logged to Google Sheets.

```
Topic (form) → Writer Agent → Reviewer Agent → ✉ Approve / Reject?
                                                   │
                         ┌──── Approve ────────────┴──── Reject + feedback ───┐
                         ▼                                                   ▼
               Log to Google Sheet                            Rework Agent → ✉ again … (repeats)
```

---

## How it works

1. **Form trigger**: you enter a **Topic** in the n8n form *LinkedIn Post Generator*.
2. **Writer Agent** drafts the post: hook, short paragraphs, professional-conversational tone, closing question or CTA.
3. **Reviewer Agent** edits the draft for clarity, hook, flow, grammar, conciseness and CTA, keeping the topic and meaning.
4. **Approval email** (Gmail *Send and Wait*) goes to the approver with the post and a form:
   - **Decision**: Approve / Reject (required)
   - **Feedback**: free text
5. **Approve** → a row is appended to Google Sheets (`Date, Topic, Final Post, Status = Approved`).
6. **Reject** → a rework agent rewrites the latest version using your feedback, and a new approval email is sent.

### Approval rounds

| Round | Rework node | Approval node | Decision | Approved → |
|---|---|---|---|---|
| 1 | — (Reviewer output) | Send message and wait for response | If | Append row in sheet |
| 2 | Rework Agent | Send message and wait for response1 | If1 | Append row in sheet1 |
| 3 | Rework Agent 2 | Approval Round 3 | If2 | Append row in sheet2 |
| 4+ (loop) | Rework Agent Loop | Loop Approval | If3 | Append row in sheet3 |

From round 4 onward, the workflow loops: **Prepare Rework Loop** (Set node) packs `Topic`, `CurrentPost` and `Feedback` and feeds them back into **Rework Agent Loop**. This repeats until you approve. **Edit Fields** does the same packing when moving from round 3 into the loop.

Each rework always starts from the **latest** version, not the original draft.

## Nodes

| Node | Type | Purpose |
|---|---|---|
| On form submission | Form Trigger | Collects the topic |
| Writer Agent | AI Agent | First draft |
| Reviewer Agent | AI Agent | Edited version |
| Rework Agent / Rework Agent 2 / Rework Agent Loop | AI Agent | Rewrites based on feedback |
| OpenAI Chat Model (×5) | `gpt-4.1-mini` | One per agent |
| Send message and wait for response / …1 / Approval Round 3 / Loop Approval | Gmail – Send and Wait (custom form) | Human approval; waits up to 24 h |
| If / If1 / If2 / If3 | IF | `{{ $json.data.Decision }} == "Approve"` |
| Edit Fields / Prepare Rework Loop | Set | Packs Topic, CurrentPost and Feedback for the loop |
| Append row in sheet (×4) | Google Sheets – Append | Logs the approved post |

## Output (Google Sheet: *LinkedIn Post Approval* → Sheet1)

| Date | Topic | Final Post | Status |
|---|---|---|---|
| 2026-10-22 00:56:00 | Agentic AI in workload automation | *(approved post text)* | Approved |

---

## Setup

See [REQUIREMENTS.md](REQUIREMENTS.md) for prerequisites.

1. Import `LinkedIn Post Approval.json` into n8n.
2. Attach credentials: **OpenAI** (5 model nodes), **Gmail** (4 approval nodes), **Google Sheets** (4 append nodes).
3. Point the 4 Sheets nodes to your spreadsheet. It needs a header row: `Date | Topic | Final Post | Status`.
4. Change the approver address (`sendTo`) in the 4 Gmail approval nodes if needed.
5. **Publish** the workflow (it's currently inactive), then open the form's **Production URL**.

## Usage

1. Open the form URL, enter a topic, and submit.
2. Wait for the approval email and click through to the response form.
3. Choose **Approve**, or **Reject** with specific feedback (e.g. "shorter, add 3 hashtags, mention Control-M").
4. Repeat until approved. The final post appears in the sheet, ready to publish on LinkedIn.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Form link returns 404 | Workflow not published / using the test URL | Publish; use the Production form URL |
| Approval link in email doesn't work | `WEBHOOK_URL` not set to the public domain | Set `WEBHOOK_URL`, restart n8n |
| Rework ignores feedback | Rejected without feedback | Always fill Feedback when rejecting |
| Post reworked after you didn't respond | 24-hour wait expired; the execution resumes with no decision, so it goes down the Reject path | Respond within 24 h, or handle the timeout (see improvements) |
| Gmail stops working after ~7 days | Google OAuth app in Testing mode | Publish the OAuth app and reconnect Gmail |
| Sheet columns misaligned | Header names differ | Match headers exactly: `Date, Topic, Final Post, Status` |

## Notes and possible improvements

- **The workflow doesn't post to LinkedIn.** It stops at logging the approved post. Add a **LinkedIn** node (Create Post) after the approval branch to publish automatically.
- **Simplify**: rounds 2 and 3 duplicate the loop logic. The Reject branch of round 1 could go straight into *Edit Fields → Rework Agent Loop*, leaving one rework agent, one approval node and one sheet node.
- **Loop limit**: there's no maximum number of rework rounds. Add a counter field in *Prepare Rework Loop* and stop after N rounds.
- **Timeout handling**: add a check for an empty `Decision`, so that a 24-hour timeout logs `Expired` instead of triggering a rework.
- **Rejected posts aren't logged.** Log every round (`Status = Rejected`, Feedback) if you want an audit trail.
- **Date timezone**: `$now` uses the instance timezone. Set `GENERIC_TIMEZONE=Asia/Kolkata` or use `$now.setZone('Asia/Kolkata')`.
