from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen3-8B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="give me the detailed explanation on {research_paper} paper with fundamentals and prerequisits",
    input_variables=["research_paper"]
)


template2 = PromptTemplate(
    template="give me a brief summary on this topic {text} max 5 lines",
    input_variables=["text"]
)
prompt1 = template1.invoke({"research_paper":'Attention is all you need'})


result1= model.invoke(prompt1)

final_result = template2.invoke({'text':result1.content})

res=model.invoke(final_result)

print(res.content)