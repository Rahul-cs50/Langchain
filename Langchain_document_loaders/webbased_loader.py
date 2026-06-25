from langchain_community.document_loaders import WebBaseLoader


url="https://sleepycat.in/products/ultima-natural-latex-mattress"
loader = WebBaseLoader(url)

text = loader.load()

print(text)