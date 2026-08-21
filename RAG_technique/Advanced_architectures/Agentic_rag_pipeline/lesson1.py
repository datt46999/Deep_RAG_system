import logging


import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated

from agent_framework import tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

load_dotenv()


endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
deployment_name = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

missing = [k for k, v in {
    "AZURE_AI_PROJECT_ENDPOINT": endpoint,
    "AZURE_AI_MODEL_DEPLOYMENT_NAME": deployment_name
}.items() if not v]

if missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Please set them as environment variables (e.g., in your .env file or shell environment)."
    )

# Create the Microsoft Foundry client
client = FoundryChatClient(
    project_endpoint=endpoint,
    model=deployment_name,
    credential=DefaultAzureCredential()
)


TRAVEL_KNOWLEDGE_BASE = {
    "Barcelona": "Barcelona is Spain's cosmopolitan capital of Catalonia. Best visited Mar-May or Sep-Nov. Known for Gaudí architecture, La Rambla, beaches. Average daily cost: $150-200.",
    "Tokyo": "Tokyo is Japan's capital, mixing ultramodern with traditional. Best visited Mar-Apr (cherry blossoms) or Oct-Nov. Known for Shibuya, temples, sushi. Average daily cost: $200-250.",
    "Paris": "Paris is France's capital and a global center for art, fashion, and culture. Best visited Apr-Jun or Sep-Oct. Known for Eiffel Tower, Louvre, cuisine. Average daily cost: $180-250.",
    "Cape Town": "Cape Town sits on South Africa's southwest tip. Best visited Nov-Mar. Known for Table Mountain, wine regions, wildlife. Average daily cost: $100-150.",
}

@tool(approval_mode = "never_require")
def search_travel_knowledge(query:Annotated[str, "The search query about the query destination"])->str:
    """
    Search the travel knowledge base on the tranvel destination
    """
    results = []
    for destination, info in TRAVEL_KNOWLEDGE_BASE.items():
        if query.lower() in destination.lower() or any(word in info.lower() for word in query.lower().split()):
            results.append(f'**{destination}**:{info}')

    return ("\n\n".join(results) if results else "NO matching destination found in the knowledge base")



"""
Building agent RAG
"""


agent = client.as_agent(
    tools = [search_travel_knowledge],
    name = "TravelRAGAgent",
    instructions = """
        You are a knowledgeable advisor. Before answering question about destinations:
        1. Alway search the travel knowledge base first
        2. Base your answer on retrieved information.
        3. If information is not in the knowledge base, say so clearly.
        4. Provide specific details like cost, best seasons, and hightlights
    """

)
response = await agent.run(
    "I'm interested in visiting somewhere with great architecture. What destination would you recomend?"
)
# print(response)


"""
Iteractive Retrieval - The maker check Pattern
"""
checker_agent =client.as_agent(
    tools = [search_travel_knowledge],
    name = "TravelRAGAgent",
    instructions ="""
        You are a meticulous travel advisor who double-checks recommendations.
        When answering travel the questions:
        1. Search relevant destinations first.
        2. For each destinaions found, search again with the destination name to get full detail.
        3. Compare the opinions using verified information.
        4. Present a final recommendation with specific cost, best travel times, and highlights.
        5. If any detail seems incomplete, search one more to confirm before responding.
    """
)
response = await agent.run(
    "I have a $175/day budget and want to travel in April. Which destinations fit my budget and timing?",
    )

print(response)