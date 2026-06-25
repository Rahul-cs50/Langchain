from langchain_core.prompts import ChatPromptTemplate

obj = ChatPromptTemplate([
    'system' , 'you are a helpful {domain} expert',
    'human' , ' explain me what is {topic}'
]
)

prompt = obj.invoke({'domain':'football','topic': 'ronaldo'})

print (prompt)