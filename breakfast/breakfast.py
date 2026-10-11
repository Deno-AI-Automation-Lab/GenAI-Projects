from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENAI_API_KEY from .env

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = (
    "Give me exactly 5 healthy breakfast ideas.\n"
    "Format rules:\n"
    "- A numbered list from 1 to 5, in the form '1. ...'\n"
    "- Exactly one line per idea, no line breaks inside an item\n"
    "- No preamble, no heading, no closing remarks, no blank lines\n"
    "- Output only the 5 numbered lines"
)

reply = llm.invoke(prompt)
print(reply.content)