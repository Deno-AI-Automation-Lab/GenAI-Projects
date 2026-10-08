"""Small LangChain app: sends several prompts to OpenAI via ChatOpenAI.

Tracing is controlled ONLY by environment variables (see .env):
    LANGSMITH_TRACING=true      -> runs are traced to LangSmith
    LANGSMITH_TRACING=false     -> nothing is traced (no code change needed)
    LANGSMITH_PROJECT=<name>    -> project the runs land in
There is no tracing code in this file.
"""
from dotenv import load_dotenv  # only loads .env into os.environ; not tracing code
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompts = [
    "In one sentence, what does a job scheduler like Control-M do?",
    "Give me a PowerShell one-liner that lists the 5 largest files in C:\\Temp.",
    "Explain the difference between latency and throughput in two sentences.",
]

for i, p in enumerate(prompts, start=1):
    answer = llm.invoke(p)
    print(f"--- Prompt {i}: {p}")
    print(answer.content)
    print()