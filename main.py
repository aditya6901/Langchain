import requests
from dotenv import load_dotenv
import os

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # Load environment variables from .env file


@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


#Creating a tool to get the current weather for a given location
@tool('get_current_weather', description="Get the current weather for a given location", return_direct=False)
def get_current_weather(location: str) -> str:

    response = requests.get(f"https://wttr.in/{location}?format=j1")

    return response.json()


@tool('locate_user', description="Locate the user based on the context")
def locate_user(runtime: ToolRuntime['Context']):
    match runtime.context.user_id:
        case "user_1":
            return "New York."
        case "user_2":
            return "Los Angeles."
        case "user_3":
            return "Chicago."
    return "User not found."


model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.3)

checkpointer = InMemorySaver()

#creating a agent which calls api of weather and returns the current weather for a given location. The agent is also humorous and cracks jokes.
#we are using get_current_weather tool to get the current weather for a given location. 
agent = create_agent(
    model=model,
    tools=[get_current_weather, locate_user],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous. You are also a weather expert and can provide accurate weather information for any location.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "thread-1"}}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the current weather like?"}
        ]
    },
    config=config,
    context=Context(user_id="user_2"),
)
 

#print(response)
#print(response["messages"][-1].content)  #Printing the content of the response

print(response['structured_response'])  #Printing the structured response
print(response['structured_response'].summary)  #Printing the structured response
print(response['structured_response'].temperature_celsius)  #Printing the structured response

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "And is this usual?"}
        ]
    },
    config=config,
    context=Context(user_id="user_2"),
)

print(response['structured_response'])  #Printing the structured response
