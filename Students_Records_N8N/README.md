# 🎓 Student Excel ETL (n8n + OpenAI + Google Drive)

An **n8n** ETL workflow that takes an uploaded student-enrolment file, cleans every row with
**OpenAI `gpt-4.1-mini`**, validates the result with a **JavaScript Code node**, and saves two Excel
reports to **Google Drive**:

- `Students_Chennai.xlsx`: students whose city is Chennai
- `Student_MailId.xlsx`: students whose email contains `@`

## Workflow

```
Form upload ─► Extract from File ─► Switch (drop empty rows) ─► Remove Duplicates (Name + Student_ID)
   ─► Basic LLM Chain ◄── OpenAI Chat Model (gpt-4.1-mini)
   ─► Loop Over Items ⇄ Code in JavaScript (parse + validate, one row at a time)
   ─► (done) ─┬─► If  City = "Chennai"   ─► Convert to File (xlsx) ─► Google Drive: Students_Chennai.xlsx
              └─► If1 Email contains "@" ─► Convert to File (xlsx) ─► Google Drive: Student_MailId.xlsx
```

| Stage | Node | What it does |
|---|---|---|
| **Extract** | CSV Upload (Form Trigger) | Web form with one file field |
| | Extract from File | Turns the uploaded file into one item per row |
| **Transform** | Switch | Drops empty rows ("Empty" output goes nowhere) |
| | Remove Duplicates | Removes rows with the same `Name` + `Student_ID` |
| | Basic LLM Chain + OpenAI | Normalises each row and returns JSON |
| | Loop Over Items + Code | Parses the AI output and re-validates every field in code, adding `Is_Valid` and `Issues` |
| **Load** | If / If1 | Splits rows into the Chennai list and the has-email list |
| | Convert to File | Builds `.xlsx` files |
| | Google Drive | Uploads both files to the `Students_Data` folder |

## Input format

Header row (column names must match):

```csv
Student_ID,Name,Email,Phone,Course,Fee_Paid,City,Enrolled_Date
s101,  manoj k ,MANOJ@GMAIL.COM,+91 98765 43210,python,YES,chennai,05/08/2026
S102,PRIYA R,priya@gmail,12345,data science,no,,2026-08-07
s101,  manoj k ,MANOJ@GMAIL.COM,+91 98765 43210,python,YES,chennai,05/08/2026
```

## Cleaning rules

**1. AI step (prompt rules)**

| Field | Rule |
|---|---|
| Name | Title Case |
| Email | `INVALID_EMAIL` if invalid, `MISSING` if empty |
| Phone | `MISSING` if fewer than 10 digits or empty |
| Course | Only Python, Machine Learning or Data Science; otherwise `MISSING` |
| Fee_Paid | yes → `true`, no → `false`, empty → `MISSING` |
| City | Title Case, empty → `UNKNOWN` |
| Enrolled_Date | `YYYY-MM-DD`, empty → `UNKNOWN` |

**2. Code step (deterministic double-check)**

The Code node checks everything again instead of trusting the AI:

- Strips ```` ```json ```` fences, double-encoded strings and trailing commas before `JSON.parse`
- **Email** must match `name@domain.tld`
- **Phone** must be a 10-digit Indian mobile number (starting 6–9); `+91` and a leading `0` are removed
- **Fee_Paid** accepts yes/y/true/paid/1 and no/n/false/unpaid/0
- **Date** accepts `YYYY-MM-DD`, `DD-MM-YYYY` and `DD/MM/YYYY`, and rejects impossible dates such as 31-02-2026
- `MISSING`/`UNKNOWN` in Student_ID, Name, Course or City is flagged
- Rows the AI returned unparseable output for become `Student_ID: PARSE_ERROR` with the raw output kept

### Output columns

`Student_ID, Name, Email, Phone, Course, Fee_Paid, City, Enrolled_Date, Is_Valid, Issues`

Example row:

| Student_ID | Name | Email | Phone | Course | Fee_Paid | City | Enrolled_Date | Is_Valid | Issues |
|---|---|---|---|---|---|---|---|---|---|
| S101 | Manoj K | manoj@gmail.com | 9876543210 | Python | true | Chennai | 2026-08-05 | true | |

## Requirements

- **n8n** (self-hosted or Cloud)
- An **OpenAI API key**
- A **Google Drive** account and destination folder

See `requirements.txt` for the full list. There are no Python packages to install.

## Setup

1. **Import:** go to **Workflows → Import from File** and choose `Student Excel ETL.json`.
2. **Credentials:** the IDs in the JSON belong to the original instance, so reconnect them:
   - **OpenAI Chat Model**: create an OpenAI credential
   - **Upload file** and **Upload file1**: create a Google Drive OAuth2 credential
3. **Drive folder:** in both upload nodes, pick your own destination folder.
4. **Run:**
   - For testing, open **CSV Upload**, click **Test step** and upload a file in the form that opens.
   - For regular use, **activate** the workflow and share the form's **Production URL**.
5. Check the folder for `Students_Chennai.xlsx` and `Student_MailId.xlsx`.

## ⚠️ Things to know

| Point | Detail |
|---|---|
| **CSV, not Excel input** | Despite the workflow's name, **Extract from File** has no operation set, so it reads CSV. To accept `.xlsx` uploads, set its operation to **Extract From XLSX** |
| Duplicate files in Drive | Each run uploads new files with the same names, and Google Drive keeps both. Add a date to the file name or use **Update file** to avoid clutter |
| Email filter is loose | `If1` only checks for `@`, so rows the Code node flagged (e.g. `priya@gmail`) still reach `Student_MailId.xlsx`. Filter on `Is_Valid` or on an empty `Issues` value for strict output |
| PARSE_ERROR rows | These have no `Email` or `City`, and with **strict** type validation the If nodes may fail on them. Switch the If nodes to loose validation if you see type errors |
| Case-sensitive Chennai match | Works because the Code node title-cases City; "Chennai " (with spaces) is also trimmed |
| Cost and speed | One OpenAI call per row, processed one at a time; large files take a while |

## Ideas for next steps

- Write invalid rows to a separate `Students_Errors.xlsx` file for review
- Add timestamps to output file names
- Email a summary (total, valid, invalid, duplicates removed) when the run finishes
- Process rows in batches of 10–20 in a single AI call to cut cost

## Project layout

```
.
├── Student Excel ETL.json   # n8n workflow export
├── requirements.txt         # prerequisites (no Python packages)
└── README.md
```
