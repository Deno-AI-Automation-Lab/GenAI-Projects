"""Agent definitions: 1 router, 5 department specialists, 1 quality observer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import Memory
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, LLMMessage, SystemMessage

from search_tool import get_search_evidence, web_search
from weather_tool import get_weather

ROUTER_NAME = "router"
OBSERVER_NAME = "quality_observer"


@dataclass(frozen=True)
class Department:
    agent_name: str
    title: str
    scope: str
    keywords: tuple[str, ...]  # fallback routing if the router's reply can't be parsed
    search_hint: str           # how this department should call web_search


DEPARTMENTS: tuple[Department, ...] = (
    Department(
        "food_agent", "Healthy Food",
        "healthy eating, nutrition, diets, recipes, ingredients, calories, meal planning",
        ("food", "diet", "eat", "recipe", "nutrition", "protein", "calorie", "meal", "vegan", "breakfast",
         "dinner", "lunch", "snack", "fruit", "vegetable"),
        "Mostly evergreen: use defaults. For new studies or guidelines use news=True, timelimit='y'.",
    ),
    Department(
        "travel_agent", "Travel",
        "destinations, itineraries, visas, flights, trains, hotels, sightseeing, travel tips",
        ("travel", "trip", "visit", "flight", "hotel", "visa", "tour", "itinerary", "vacation", "places",
         "train", "airport", "holiday", "sightseeing"),
        "Visa rules, fares, closures and events change: use timelimit='m' (news=True for advisories).",
    ),
    Department(
        "exercise_agent", "Exercise",
        "workouts, fitness plans, running, yoga, strength training, stretching, gym routines",
        ("exercise", "workout", "gym", "run", "yoga", "fitness", "cardio", "treadmill", "stretch", "muscle",
         "walking", "weight loss"),
        "Mostly evergreen: use defaults.",
    ),
    Department(
        "movies_agent", "Movies",
        "films in any language (Tamil, Hindi, English...), now showing in theatres, releases, reviews, cast, "
        "box office, OTT/streaming, songs, trailers, recommendations",
        ("movie", "film", "cinema", "actor", "actress", "release", "netflix", "ott", "box office", "director",
         "trailer", "theatre", "theater", "kollywood", "bollywood", "tollywood", "imdb", "prime video", "hotstar"),
        "Now showing / trending / box office: news=True, timelimit='w' with short keywords such as "
        "'Tamil box office collection' or 'Tamil movie release this Friday' (news about collections names the "
        "films actually running). Upcoming: 'upcoming Tamil movies release date', timelimit='m'. "
        "One film's details: '<film name>' with news=True, timelimit='m'.",
    ),
    Department(
        "weather_agent", "Weather",
        "current weather, forecasts, temperature, rain, monsoon, cyclone, air quality, climate for a place",
        ("weather", "rain", "temperature", "forecast", "humid", "monsoon", "climate", "sunny", "storm", "aqi",
         "cyclone", "heat", "cold"),
        "Call get_weather('<place>') FIRST for current conditions, temperature, rain chance and forecast: it "
        "returns live measured data. Use web_search (news=True, timelimit='w') only for alerts, cyclone or "
        "IMD warnings, or if get_weather fails. Report the numbers exactly as given (°C, %, mm, km/h).",
    ),
)

SPECIALIST_NAMES: tuple[str, ...] = tuple(d.agent_name for d in DEPARTMENTS)


def _router_prompt() -> str:
    menu = "\n".join(f"- {d.agent_name}: {d.title} ({d.scope})" for d in DEPARTMENTS)
    return f"""You are the front-desk router of a customer support desk. You never answer questions yourself.
Departments:
{menu}

Read the customer's latest question. If it is a follow-up ("this movie", "it", "there", "tomorrow"),
resolve what it refers to using the MOST RECENT exchange in the conversation memory, not older ones.
Rewrite as little as possible: only replace those reference words. NEVER add names, films, places,
dates (not even today's date) or conditions the customer did not mention (e.g. do not turn "Rajinikanth's upcoming movie"
into "Rajinikanth's upcoming movie after Jailer 2", and do not append "as of <date>").
If the question is already complete, copy it unchanged.
When unsure, pick the closest department. Anything about films, actors, theatres or OTT is movies_agent.

Reply with exactly two lines and nothing else:
ROUTE: <agent_name>
QUESTION: <the customer's question rewritten as a complete standalone question, with every
"it/this/that/there" replaced by the actual name, place or thing it refers to>

Example: after an exchange about Rajinikanth's next film "Dharman", the question
"in this movie who is the director" becomes
ROUTE: movies_agent
QUESTION: Who is the director of Rajinikanth's upcoming film Dharman?

Only if the question clearly has nothing to do with any department, reply with one line:
OUT_OF_SCOPE: <one polite sentence saying we cover healthy food, travel, exercise, movies and weather>"""


def _specialist_prompt(d: Department) -> str:
    return f"""You are the {d.title} specialist on a customer support desk.
Your scope: {d.scope}.
Today's date is given at the top of the customer's message; treat it as "now".
Answer the standalone QUESTION written by the router (it already resolves follow-ups).

Rules:
1. Always call a tool before answering. Guidance for your department: {d.search_hint}
   Use short keyword queries (3-6 words) built ONLY from the QUESTION's own words. Never add film names
   or topics from earlier conversation that the QUESTION doesn't contain: they pull in results about
   the wrong subject. Do NOT add month/year or "now"/"latest" when you set timelimit.
   If the results don't contain what was asked, you MUST search again with different wording
   (synonyms, fewer words, e.g. "Rajinikanth next film", "Rajinikanth new movie announcement")
   and make 3 searches in total before you say it was not found.
2. Every fact (name, date, number, title) must be stated in a search result, and must be about the
   film/place/topic in the QUESTION. Results often mention several films or people: never move a fact
   from one subject to another. Conversation memory is only for understanding the question; it is
   NOT a source of facts. Never fill gaps from your own knowledge.
3. If, after 3 searches, the specific thing asked (e.g. the director) is still not in the results, say
   plainly in the first bullet that it was not found in current sources. That is better than guessing.
   Use the other bullets only for facts that are directly about the QUESTION's subject; never pad with
   unrelated news about other films or topics.
4. Reply in exactly 5 bullet lines, each starting with "- ", under 22 words, and ending with the id of
   the supporting result, e.g. "- Nayanthara has joined the cast. [S2]". Cite one or two ids
   like [S2] or [S2, S4], never ranges. A bullet that says something was not found ends with [none].
   No greeting, heading, intro, closing line, raw URLs or pasted snippets.
5. For time-sensitive questions prefer the newest dated results and mention the date when useful.
6. If quality_observer replies REVISE, fix exactly what it flagged (remove or correct unsupported
   bullets) and reply again in the same format."""


_OBSERVER_PROMPT = """You are the quality observer (fact checker) on a customer support desk.
You watch the team and never answer the customer.
The exact evidence the specialist received (searches made + every result with its [S#] id) is
attached as the LAST message of your context. Always judge against that evidence.

Check the specialist's latest reply against the router's QUESTION and the evidence:
  (a) every name, date, number and title in a bullet is stated in the cited [S#] result,
      and is about the same subject as the QUESTION (not a different film/person/place);
  (b) it answers the QUESTION that was asked;
  (c) exactly 5 bullets, each ending with a citation like [S3] or [none];
  (d) no padding: bullets about other topics that don't help answer the QUESTION are wrong.
Rules:
- If live data (e.g. Open-Meteo weather) is in the evidence, the reply must use it, not say "not found".
- If the reply says the answer was not found and fewer than 3 searches were made, reply REVISE and
  suggest two simpler, different search queries. After 3 searches, "not found" is acceptable.
- Do not ask for "more current" information when the facts are supported by the evidence.

Reply with exactly one line:
APPROVED
or
REVISE: <which bullet is wrong and why, e.g. "Bullet 1: director name is not in [S2], which is about another film">"""


class EvidenceContext(UnboundedChatCompletionContext):
    """Observer's model context: always appends the current search evidence.

    In a group chat, a specialist's tool results stay private to it, so without this
    the observer could only judge the bullets' format. Injecting the evidence on every
    call is deterministic, unlike hoping the observer remembers to call a tool.
    """

    async def get_messages(self) -> List[LLMMessage]:
        messages = list(await super().get_messages())
        evidence = get_search_evidence()
        return messages + [SystemMessage(content="EVIDENCE for the current question:\n" + evidence)]


def build_agents(
    model_client: ChatCompletionClient, memory: Memory
) -> tuple[AssistantAgent, list[AssistantAgent], AssistantAgent]:
    router = AssistantAgent(
        name=ROUTER_NAME,
        model_client=model_client,
        description="Reads the customer's question and decides which department agent should answer it.",
        system_message=_router_prompt(),
        memory=[memory],
    )

    specialists = [
        AssistantAgent(
            name=d.agent_name,
            model_client=model_client,
            description=f"{d.title} department specialist: {d.scope}.",
            system_message=_specialist_prompt(d),
            tools=[get_weather, web_search] if d.agent_name == "weather_agent" else [web_search],
            reflect_on_tool_use=True,  # write an answer from the results, not a raw dump
            max_tool_iterations=3,      # up to 3 searches before answering
            memory=[memory],
        )
        for d in DEPARTMENTS
    ]

    observer = AssistantAgent(
        name=OBSERVER_NAME,
        model_client=model_client,
        description="Fact-checks the specialist's answer against the search evidence; approves or asks for one revision.",
        system_message=_OBSERVER_PROMPT,
        model_context=EvidenceContext(),
    )
    return router, specialists, observer
