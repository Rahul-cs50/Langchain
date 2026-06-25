from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen3-0.6B',
    task = 'text-generation',
    pipeline_kwargs=dict(
        temperature=0.8,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)
result  = model.invoke("Captial of Kazakisthan??")
print(result.content)