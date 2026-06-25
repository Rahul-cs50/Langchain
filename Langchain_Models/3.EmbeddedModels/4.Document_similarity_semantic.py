from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
docs= [
    "Hello  i am Rahul!",
    "I love to eat Pizza",
    "I am from India!",
    "India is very big country",
    "Capital of India is New Delhi"
]
user_query = " Rahul Love to eat Pizza! in India"
emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
doc_vec = emb.embed_documents(docs)
user_vec = emb.embed_query(user_query)

sim_vec=cosine_similarity([user_vec],Y=doc_vec)[0]
index,score = sorted(list(enumerate(sim_vec)),key=lambda x:x[1])[-1]

print(user_query)
print(docs[index])
print("similarity: ",str(score))