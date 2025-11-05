import asyncio
import logging
import os

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv #Used for loading .env files
from openai import AsyncOpenAI

logging.basicConfig(level=logging.WARNING)
# Disable tracing since we're not connected to a supported tracing provider
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

async_credential = None

client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")


agent = Agent(
    name="Spanish tutor",
    instructions="You are a Spanish tutor. Help the user learn Spanish. ONLY respond in Spanish.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


async def main():
    result = await Runner.run(agent, input="hi how are you?")
    print(result.final_output)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
