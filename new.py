import requests
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
load_dotenv()  # Load environment variables from .env file

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)


response = model.invoke("What is python?")

print(response)
print(response["messages"][-1].content)  