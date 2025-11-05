import asyncio
import os

from agents import (Agent, 
    OpenAIChatCompletionsModel, 
    Runner, 
    function_tool, 
    set_tracing_disabled, 
    SQLiteSession)

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)

# Adding short term in-memory async credential management
session = SQLiteSession("conversation_history")

async_credential = None

API_HOST = os.getenv("API_HOST", "openai")

if API_HOST == "github":
    client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
elif API_HOST == "openai":
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")
else:
    raise ValueError(f"Unsupported API_HOST: {API_HOST}")

@function_tool
def aerialist(city: str) -> str:
    if city =='Bogota':
        return {
        "city": city,
        "Studio": "Liberte",
        "elements": "Hoop and Silks",
        "times": "Tuesdays and Thursdays at 6 PM",
        "instructor": "Maria Gonzalez",
    }
    else:      return {
        "city": city,
        "Studio": "AeroFit",
        "elements": "Trapeze and Straps",
        "times": "Wednesdays and Fridays at 7 PM",
        "instructor": "Carlos Ramirez",
    }

agent = Agent(
    name="Aerialist agent",
    instructions="Return the Studio name and elements for a given city",
    tools=[aerialist],
)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish. Return the information in bullets.",
    tools=[aerialist],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English. Return the information in bullets.",
    tools=[aerialist],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request. Do not add any additional information.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

#input_user = "Cuales son las clases en Bogota de elementos aereo en comparacion a Medellin?"
input_user = "In Bogota, what are the classes for aerial elements?"

async def main():
    result = await Runner.run(triage_agent, input=input_user,  session=session)
    print(result.final_output)
    history = await session.get_items()
    print(history)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
