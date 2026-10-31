"""Interactive customer support chatbot (AutoGen multi-agent).

Run:  python main.py            (live trace of what each agent does)
      python main.py --quiet    (answers only)

Chat commands:
  /remember <fact>   save a preference, e.g. "/remember I live in Chennai"
  /memory            show what is in memory and where it is saved
  /history           show the last few saved turns
  /forget            clear memory
  /exit              quit
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import settings
from memory_store import ConversationMemory
from observability import TraceObserver
from team import SupportDesk

BANNER = """
╭───────────────────────────────────────────────────────────╮
│  Customer Support Desk — Healthy Food · Travel · Exercise │
│  Movies · Weather     /memory /history /forget /exit      │
╰───────────────────────────────────────────────────────────╯"""


async def chat_loop(desk: SupportDesk, memory: ConversationMemory) -> None:
    print(BANNER)
    while True:
        try:
            question = (await asyncio.to_thread(input, "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            break
        if question.lower().startswith("/remember "):
            await memory.remember_fact(question[len("/remember "):].strip())
            print("Saved.")
            continue
        if question.lower() == "/memory":
            print(f"{len(memory.facts)} fact(s), {len(memory.turns)} turn(s) saved in {memory.path}")
            for f in memory.facts:
                print(f"  - {f}")
            continue
        if question.lower() == "/history":
            print(memory.history_text() or "No saved turns yet.")
            continue
        if question.lower() == "/forget":
            await memory.clear()
            print("Memory cleared.")
            continue

        try:
            answer = await desk.ask(question)
        except Exception as exc:  # keep the chat alive on API/network errors
            logging.getLogger("support_bot").exception("Run failed")
            print(f"Sorry, something went wrong ({type(exc).__name__}). Please try again.")
            continue

        print(f"\n[{answer.agent}]")
        if answer.resolved_question != question:
            print(f"\033[2m(understood as: {answer.resolved_question})\033[0m")
        print(answer.text)
        if answer.sources:
            print("\033[2mSources: " + "  ".join(answer.sources[:3]) + "\033[0m")
        print(
            f"\033[2m· {answer.status} · revisions {answer.revisions} · searches {answer.tool_calls} · "
            f"tokens {answer.prompt_tokens}+{answer.completion_tokens} · {answer.elapsed_s}s\033[0m"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGen customer support chatbot")
    parser.add_argument("--quiet", action="store_true", help="hide the live agent trace")
    parser.add_argument("--ask", help="answer one question and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings.validate()

    model_client = OpenAIChatCompletionClient(
        model=settings.openai_model, api_key=settings.openai_api_key, temperature=settings.temperature
    )
    memory = ConversationMemory(settings.memory_file, settings.memory_max_entries)
    observer = TraceObserver(settings.log_dir, verbose=not args.quiet)
    print(f"Memory: {len(memory.facts)} fact(s), {len(memory.turns)} turn(s) loaded from {memory.path}")
    try:
        desk = SupportDesk(model_client, memory, settings, observer)
        if args.ask:
            answer = await desk.ask(args.ask)
            print(f"\n[{answer.agent}]\n{answer.text}")
            if answer.sources:
                print("Sources: " + "  ".join(answer.sources[:3]))
        else:
            await chat_loop(desk, memory)
    finally:
        await memory.close()
        await model_client.close()  # always release the HTTP client
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
