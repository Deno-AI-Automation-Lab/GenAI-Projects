"""
HR Assistant agent built with AutoGen (AgentChat API, v0.4+).

Setup:
    pip install -U "autogen-agentchat" "autogen-ext[openai]" python-dotenv

Create a .env file next to this script containing:
    OPENAI_API_KEY=sk-...

Run:
    python hr_agent.py
"""

import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    # Load OPENAI_API_KEY from the .env file into the environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

    # Model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        # The agent
        hr_agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are a Junior Python Developer",
        )

        task = (
            "Write a 5-line job description in bullet points "
            "for a Junior Python Developer"
        )

        # Stream the conversation to the terminal and print the reply
        await Console(hr_agent.run_stream(task=task))
    finally:
        # Always close the client, otherwise the script can hang on exit
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())