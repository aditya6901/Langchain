from dotenv import load_dotenv

load_dotenv()


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


from langchain_core.tools.retriever import create_retriever_tool

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "I love apples",
    "I enjoy oranges",
    "I think pears taste very good",
    "I hate bananas",
    "I love Linux",
    "I dislike Windows",
    "I despise Mangoes",
    "I dislike rasberries"
]



vectorstore = FAISS.from_texts(texts, embeddings)

#print(vectorstore.similarity_search("Apples are my favorite fruit", k=3))
#print(vectorstore.similarity_search("Linux is a powerful operating system", k=3))


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='kb_search',
    description="Search the small products/fruits knowledge base"
)



model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.3)

#creating a agent which calls api of weather and returns the current weather for a given location. 
#we are using get_current_weather tool to get the current weather for a given location. 
agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant, for question about laptops and fruits. You are also a expert and can provide accurate information for any question about laptops and fruits."
                    "first call the kb_search tool to search the knowledge base for relevant information, yhen answer succinctly. Maybe you have to use it multiple times to get all the relevant information and answer"
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}
    ]  })


print(result["messages"][-1].content)  #Printing the content of the response