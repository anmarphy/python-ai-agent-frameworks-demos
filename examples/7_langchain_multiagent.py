import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from rich import print
from rich.logging import RichHandler

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


logging.basicConfig(level=logging.WARNING, 
                    format="%(message)s", 
                    datefmt="[%X]", 
                    handlers=[RichHandler()])
logger = logging.getLogger("aerialist_advisor")

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

@tool
def get_schedule(city: str) -> str:
    """Returns the classes information for a given city."""
    logger.info(f"Getting classes information for {city}")
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


@tool
def get_current_date() -> str:
    """Gets the current date from the system and returns as a string in format YYYY-MM-DD."""
    logger.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


classes_agent = create_agent(
    model=model,
    system_prompt= ("You help users to get the information about aerial classes. "
                    "Return the Studio name and the information of the classes for a given city in bullets. "
                    "Use the current date to suggest upcoming classes."
                    ),
    tools=[get_schedule, get_current_date],
)

@tool
def information_classes(query: str) -> str:
    """Plan a weekend based on user query and return the final response."""
    logger.info("Tool: classes_agent invoked")
    response = classes_agent.invoke({"messages": [HumanMessage(content=query)]})
    final = response["messages"][-1].content
    return final

spanish_agent = create_agent(
    system_prompt="You only speak Spanish. Return the information in bullets.",
    tools=[information_classes],
    model=model,
)

@tool
def information_spanish(query: str) -> str:
    """Process and retrieve the information in Spanish."""
    logger.info("Tool: spanish_agent invoked")
    response = spanish_agent.invoke({"messages": [HumanMessage(content=query)]})
    final = response["messages"][-1].content
    return final

english_agent = create_agent(
    system_prompt="You only speak English. Return the information in bullets.",
    tools=[information_classes],
    model=model,
)

@tool
def information_english(query: str) -> str:
    """Process and retrieve the information in English."""
    logger.info("Tool: spanish_agent invoked")
    response = spanish_agent.invoke({"messages": [HumanMessage(content=query)]})
    final = response["messages"][-1].content
    return final

supervisor_agent = create_agent(
    model=model,
    system_prompt=("Retrieve information about aerial classes based on user queries. " "Use the tool to get the information and return it to the user."),
    tools=[information_spanish, information_english],
)


def main():
    response = supervisor_agent.invoke({"messages": [{"role": "user", "content": "hii what aerial classes can I do in Bogota?"}]})
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()