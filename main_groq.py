import requests
from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_groq import ChatGroq   # <-- changed import

load_dotenv()  # Load environment variables from .env file

#Creating a tool to get the current weather for a given location
@tool('get_current_weather', description="Get the current weather for a given location", return_direct=False)
def get_current_weather(location: str) -> str:

    response = requests.get(f"https://wttr.in/{location}?format=j1")

    return response.json()


#creating a agent which calls api of weather and returns the current weather for a given location. The agent is also humorous and cracks jokes.
#we are using get_current_weather tool to get the current weather for a given location. 
agent = create_agent(
    model=ChatGroq(model="llama-3.3-70b-versatile"),   # <-- changed model
    tools=[get_current_weather],
    system_prompt="You are a helpful weather assistant, who always cracks jokes and is humorous. You are also a weather expert and can provide accurate weather information for any location.",
)


response = agent.invoke({
    'messages': [

        {"role": "user", "content": "What is the current weather in Mumbai?"}

    ]
})
 

print(response)
print(response["messages"][-1].content)  #Printing the content of the response