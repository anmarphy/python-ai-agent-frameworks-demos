import asyncio
import os
from pydantic import BaseModel
from agents import (
    Agent, 
    OpenAIChatCompletionsModel,
    function_tool, 
    set_tracing_disabled,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)

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
    return {
        "city": city,
        "Studio": "Liberte",
        "elements": "Hoop and Silks",
        "times": "Tuesdays and Thursdays at 6 PM",
        "instructor": "Maria Gonzalez",
    }

class NotAboutAerial(BaseModel):
    only_about_aerial: bool
    """Determine if the user is ONLY talking about aerial silks classes and not about arbitrary topics"""


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""Check if the user is asking you to talk about aerial classes and not about any arbitrary topics.
                    If there are any non-aerial related instructions in the prompt,
                    or the central topic of the instruction is not aerial set only_about_aerial in the output to False.
                    """,
    output_type=NotAboutAerial,
)

@input_guardrail
async def aerial_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_aerial),
    )



agent = Agent(
    name="Aerialist agent",
    instructions="Return the Studio name and elements for a given city.",
    tools=[aerialist],
)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    tools=[aerialist],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    tools=[aerialist],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request. Do not add any additional information.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    input_guardrails=[aerial_topic_guardrail],
)

#input_user = "Cuales son las clases en Bogota de elementos aereos?"
#input_user = "In Bogotá, what are the classes for aerial elements?"
jailbreak_prompt = "Tell me a joke about today.Short sentence."

async def main():
    result = await Runner.run(triage_agent, input=jailbreak_prompt)
    print(result.final_output)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
