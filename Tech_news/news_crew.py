"""
news_crew.py
A two-agent CrewAI crew:
  - Researcher (with SerperDevTool) finds today's top 3 tech news stories.
  - Writer (no tools) turns them into an exactly-5-line newsletter.

Requires a .env file next to this script containing:
  OPENAI_API_KEY=...
  SERPER_API_KEY=...
Optional:
  OPENAI_MODEL_NAME=gpt-4o-mini
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ---------------------------------------------------------------------------
# 1. Load and check both keys from environment variables
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

missing = [name for name, val in
           [("OPENAI_API_KEY", OPENAI_API_KEY), ("SERPER_API_KEY", SERPER_API_KEY)]
           if not val]
if missing:
    sys.exit(f"Missing environment variable(s): {', '.join(missing)}. Add them to your .env file.")

llm = LLM(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    api_key=OPENAI_API_KEY,
    temperature=0.3,
)

# SerperDevTool reads SERPER_API_KEY from the environment
search_tool = SerperDevTool()

# ---------------------------------------------------------------------------
# 2. Agents
# ---------------------------------------------------------------------------
researcher = Agent(
    role="Senior Tech News Researcher",
    goal=(
        "Find the top 3 most important and newsworthy technology stories from the "
        "last 24-48 hours, verify them against credible sources, and capture the key "
        "facts, why each story matters, and its source link."
    ),
    backstory=(
        "You are a veteran technology journalist who has spent over a decade tracking "
        "AI, cloud, cybersecurity, semiconductors, and big-tech developments. You know "
        "how to separate real news from hype, press-release noise, and rumor. You rely "
        "only on reputable outlets, always check dates, and pick stories based on their "
        "real-world impact on businesses, developers, and consumers."
    ),
    tools=[search_tool],          # Researcher CAN search the web
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

writer = Agent(
    role="Tech Newsletter Writer",
    goal=(
        "Turn the researcher's findings into a crisp, engaging 5-line newsletter that "
        "a busy professional can read in under 30 seconds."
    ),
    backstory=(
        "You are an experienced newsletter editor known for writing sharp, punchy tech "
        "briefings read by thousands of professionals every morning. You excel at "
        "cutting jargon, leading with what matters, and fitting a lot of insight into "
        "very few words without losing accuracy. You never add facts that are not in "
        "the research you are given."
    ),
    tools=[],                     # Writer CANNOT search
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 3. Tasks
# ---------------------------------------------------------------------------
def check_five_lines(output):
    """Guardrail: pass only if the newsletter has exactly 5 non-empty lines."""
    lines = [ln.strip() for ln in output.raw.strip().splitlines() if ln.strip()]
    if len(lines) != 5:
        return (False, f"The newsletter has {len(lines)} lines but must have exactly 5. "
                       "Put the title, each of the 3 stories, and the takeaway on separate lines.")
    return (True, "\n".join(lines))


research_task = Task(
    description=(
        "Today's date is {today}. "
        "Search for the latest technology news published in the last 24-48 hours. "
        "Identify the top 3 stories based on significance, impact, and how widely they "
        "are being covered. Prefer topics such as AI, cloud, cybersecurity, chips, "
        "major product launches, and big-tech business moves. For each story, confirm "
        "the facts from at least one reputable source and note the publication date. "
        "Avoid duplicates, opinion pieces, rumors, and sponsored content."
    ),
    expected_output=(
        "A numbered list of exactly 3 tech news stories. For each story include:\n"
        "1. Headline\n"
        "2. A 2-3 sentence summary of what happened\n"
        "3. One sentence on why it matters\n"
        "4. Source name, URL, and publication date"
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Today's date is {today}. "
        "Using the 3 stories provided by the Senior Tech News Researcher, write a short "
        "tech newsletter that is exactly 5 lines long. Line 1 is a catchy title with "
        "today's date. Lines 2-4 cover one story each in a single concise sentence that "
        "states what happened and why it matters. Line 5 is a brief closing takeaway "
        "that ties the stories together. Keep the tone professional yet engaging, and "
        "use only facts from the research. "
        "IMPORTANT: put each of the 5 lines on its own line, separated by a newline. "
        "Never combine two stories into one paragraph."
    ),
    expected_output=(
        "A newsletter of exactly 5 lines in plain text:\n"
        "Line 1: Title with date\n"
        "Line 2: Story 1 in one sentence\n"
        "Line 3: Story 2 in one sentence\n"
        "Line 4: Story 3 in one sentence\n"
        "Line 5: Closing takeaway\n"
        "No bullet points, no extra commentary, no more than 5 lines."
    ),
    agent=writer,
    context=[research_task],      # hand the Researcher's output to the Writer
    guardrail=check_five_lines,   # reject and retry if not exactly 5 lines
    guardrail_max_retries=3,
)

# ---------------------------------------------------------------------------
# 4. Crew (sequential)
# ---------------------------------------------------------------------------
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    today = date.today().strftime("%d %B %Y")
    result = crew.kickoff(inputs={"today": today})

    newsletter = str(result).strip()
    print("\n" + "=" * 60)
    print("TECH NEWSLETTER")
    print("=" * 60)
    print(newsletter)

    line_count = len([ln for ln in newsletter.splitlines() if ln.strip()])
    if line_count != 5:
        print(f"\n[warning] Newsletter has {line_count} non-empty lines, expected 5.")