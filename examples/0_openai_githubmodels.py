import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a positive helpful assistant.",
        },
        {
            "role": "user",
            "content": "Give a motivational quote for starting a new project.",
        },
    ],
    model=os.getenv("GITHUB_MODEL", "gpt-4o"),
)
print(response.choices[0].message.content)
