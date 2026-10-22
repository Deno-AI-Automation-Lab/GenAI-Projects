"""
Travel Planner - a simple CrewAI script.
The user enters a city, and the Travel Planner agent suggests 3 places to visit there.

Requirements:
    pip install crewai python-dotenv

.env file (same folder as this script):
    OPENAI_API_KEY=sk-...
    MODEL=gpt-4o-mini
"""

import os

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load OPENAI_API_KEY and MODEL from the .env file
load_dotenv()

llm = LLM(model=os.getenv("MODEL"))

# Agent: Travel Planner
travel_planner = Agent(
    role="Seasoned Travel Guide with 15+ years of global travel experience",
    goal=(
        "Recommend the top 3 must-visit places in {city}, explain why each matters, "
        "and suggest the best time of day to visit each one."
    ),
    backstory=(
        "You have explored hundreds of cities across every continent. You know each "
        "place's cultural and historical significance, and you know when to visit to "
        "avoid crowds, catch the best light, beat the heat, or see local life at its "
        "liveliest. Travellers trust you for concise, practical advice."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# Task: List Places
list_places = Task(
    name="List Places",
    description=(
        "Identify the top 3 places to visit in {city}. For each place, give one reason "
        "it's worth visiting and the best time of day to go (morning, afternoon, evening, "
        "or night) with a short justification."
    ),
    expected_output=(
        "Exactly 3 bullet points, one line each, in this format: "
        "• <Place name> – <why it's worth visiting> – Best time: <time of day> (<reason>). "
        "No introduction or closing text."
    ),
    agent=travel_planner,
)

# Crew: one agent, one task
crew = Crew(
    agents=[travel_planner],
    tasks=[list_places],
    verbose=True,
)


if __name__ == "__main__":
    city = input("Enter a city: ").strip()
    while not city:
        city = input("City cannot be empty. Enter a city: ").strip()

    result = crew.kickoff(inputs={"city": city})

    print(f"\nTop 3 places to visit in {city}:\n")
    print(result)