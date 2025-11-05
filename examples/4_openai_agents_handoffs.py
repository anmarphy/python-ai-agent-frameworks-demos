import asyncio
import os

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

async_credential = None
client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")


@function_tool
def get_weather(city: str) -> str:
    return {
        "city": city,
        "temperature": 72,
        "description": "Sunny",
    }


agent = Agent(
    name="Weather agent",
    instructions="You can only provide weather information.",
    tools=[get_weather],
)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    tools=[get_weather],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    tools=[get_weather],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás? ¿Puedes darme el clima para San Francisco CA?")
    print(result.final_output)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
