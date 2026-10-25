# FiberNet Broadband – AI Customer Support Agent

CrewAI multi-agent support assistant grounded in the **FiberNet Broadband Customer Support Troubleshooting Handbook** (PDF) through a LangChain RAG tool backed by FAISS, with a Streamlit chat UI.

## Architecture

```
Streamlit chat (app.py)
   │  customer message + sidebar details (LEDs, affected devices…) + chat history
   ▼
CrewAI crew (crew/support_crew.py) – sequential
   1. Broadband Triage Analyst      → impact, fault domain, priority P1–P4, ticket tags
   2. FTTH Troubleshooting Engineer → ordered steps, NOC/Field escalation decision
   3. Customer Support Specialist   → customer reply + case note (Pydantic SupportResponse)
   │  every agent can call ↓
   ▼
fibernet_handbook_search  (rag/rag_tool.py – CrewAI BaseTool)
   ▼
LangChain FAISS index (rag/vectorstore.py)
   PyPDFLoader → header cleanup + section tagging → RecursiveCharacterTextSplitter
   → OpenAIEmbeddings (text-embedding-3-small) → FAISS (saved to vectorstore/)
```

## Project layout

```
fibernet-support-agent/
├── app.py                 # Streamlit UI
├── config.py              # loads .env, model names, paths, chunk settings
├── crew/
│   ├── schemas.py         # SupportResponse (structured final output)
│   └── support_crew.py    # agents, tasks, crew
├── rag/
│   ├── vectorstore.py     # PDF → chunks → embeddings → FAISS (build/load)
│   └── rag_tool.py        # CrewAI tool wrapping the FAISS search
├── data/FiberNet_Broadband_Network_Troubleshooting_Handbook_20_Pages.pdf
├── requirements.txt
└── .env.example
```

## Setup (Windows PowerShell)

```powershell
cd fibernet-support-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env      # then put your OPENAI_API_KEY in .env

python -m rag.vectorstore   # builds the FAISS index once (optional – app auto-builds)
streamlit run app.py
```

Linux/macOS: `source .venv/bin/activate` and `cp .env.example .env`.
CrewAI needs Python 3.10 – 3.13.

## Using it

1. Optionally fill the sidebar (affected devices, ONT Power/PON/LOS LEDs, router WAN LED, pattern, plan).
2. Type the customer's issue. The reply shows:
   - customer-facing steps,
   - fault domain, priority and ticket tags,
   - follow-up questions, escalation note (NOC / Field technician), internal case note,
   - the exact handbook passages (page + section) the agents retrieved.
3. Follow-ups in the same chat keep context; **New case** clears it.
4. Replace the PDF in `data/` (or set `PDF_PATH` in `.env`) and press **Rebuild index** to use another handbook.

## Sample questions

- Internet is down on all devices; the fiber box shows a blinking red LOS light since the power cut.
- Speed test shows 94 Mbps on my 300 Mbps plan, even with the laptop on cable.
- Wi-Fi says connected but no internet on my phone only; laptop is fine.
- Connection drops every 15 minutes in the evening, Wi-Fi and wired both.
- Online gaming lag spikes when someone is uploading videos.

## Notes

- The OpenAI key is read only from `.env`; `.env` and the `vectorstore/` folder are git-ignored.
- `allow_dangerous_deserialization=True` is used only to load the FAISS index this app built itself.
- Each question runs 3 agents with several tool calls, so expect ~15–40 s per answer with `gpt-4o-mini`. Switch `LLM_MODEL` in `.env` for a stronger model.
- Turn on **Show agent reasoning in terminal** to watch CrewAI's step-by-step logs.
