from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=67)

documents = [
    "Hello i am rahul",
    "I am from india fellas!",
    "I love LLMs"
]
result = embedding.embed_documents(documents)

print(str(result))