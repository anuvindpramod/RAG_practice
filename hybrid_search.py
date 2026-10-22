from langchain.retrivers import EnsembleRetriever, BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import argparse

load_dotenv()

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

#convert to Document object for langchain

documents=[Document(
    page_content=chunk,
    metadata={
        "source":f"chunk {i}" for i,chunk in enumerate(chunks)
    }
)]

# print("sample data")
# for i,chunk in enumerate(chunks,1):
#     print(f"{i}. chunk: {chunk}")

# 1 . Vector Search(semantic search/dense retrieval
print("setting up vector search ...")
emdedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
vector_store=Chroma.from_documents(
    embedding=emdedding_function,
    collection_metadata={"hnsw:space":"cosine"},
    documents=documents
)
vector_retriver=vector_store.as_retriever(
    search_kwargs={"k":2}
)
test_query = "space exploration company"
print(f"testing {test_query}")

test_doc=vector_retriver.invoke(test_query)
for doc in test_doc:
    print(f"Found {doc.page_content}")


#2. BM25 retriever (Key search/sparse retrival)
print("setting up BM25 retriever...")
bm25_retriver=BM25Retriever.from_documents(documents)
bm25_retriver.k=3
test_query="tesla"
print(f"Testing the bm25_retiever with the query:{query}")
test_docs=bm25_retriver.invoke(test_query)
for doc in test_doc[:2]:
    print(f"found the docs{doc.page_content}")


#3. Hybrid retriever
print("Setting up hybrid retriver")
hybrid_retriver=EnsembleRetriever(
    retrivers=[vector_retriver,bm25_retriver],
    weights=[0.5,0.5]
)
