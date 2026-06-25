from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader


loader = DirectoryLoader(
    path= "books",
    glob= '*.pdf',
    loader_cls=PyPDFLoader
    )

text = loader.load()
print((text[100].page_content))