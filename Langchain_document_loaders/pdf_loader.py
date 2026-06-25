from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("dl-curriculum.pdf")


llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="what fundamental topics are missing from this \n {text}??",
    input_variables=['text']

)
text = loader.load()

parser  = StrOutputParser()

chain = prompt| model | parser

print(chain.invoke({'text': text}))

