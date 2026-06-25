from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import grandalf
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template="give me detailed information about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="give me 5 line summary about the following text \n  {text}",
    input_variables=["text"]
)
parser = StrOutputParser()

chain = prompt1|model|parser|prompt2|model|parser

result = chain.invoke({'topic': 'chocolate'})

print(result)

chain.get_graph().print_ascii()