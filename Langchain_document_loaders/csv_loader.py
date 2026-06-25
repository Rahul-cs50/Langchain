from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('Social_Network_Ads.csv')

text = loader.load()
print((text[399]))