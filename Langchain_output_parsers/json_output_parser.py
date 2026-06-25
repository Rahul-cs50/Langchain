from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen3-8B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template1 = PromptTemplate(
    template="give me summary on {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template1|model|parser

result = chain.invoke({'topic':'Attention is all you need'})

print(result)