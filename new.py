import requests
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv()  # Load environment variables from .env file

from langchain_google_genai import ChatGoogleGenerativeAI

#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

model = init_chat_model(
    model="gemini-2.5-flash-lite",
    model_provider="google_genai",
    temperature=0.1,
)

conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is python?"),
    AIMessage(content="Python is a high-level, interpreted programming language known for its simplicity and readability."),
    HumanMessage(content="When was it released?"),
]

#invoking on a string prompt
#response = model.invoke("What is python?")

#invoking on conversation messages
response = model.invoke(conversation)

print(response)
print(response.content )