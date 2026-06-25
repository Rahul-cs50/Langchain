from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt",encoding="utf-8")

text = loader.load()
print(text)
print(len(text))
print(text[0])
print(text[0].page_content)
print(text[0].metadata)