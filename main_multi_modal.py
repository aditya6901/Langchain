from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from transformers import pipeline

load_dotenv()  # Load environment variables from .env file


model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.3)

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the contents of this image.'},
        {'type': 'image', 'url': 'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d'}
    ]
}

response = model.invoke([message])

print(response.content)