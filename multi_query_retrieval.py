from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()

persistant_directory="db2/chroma_db"
embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
llm=ChatOpenAI(model="gpt-4o",temperature=0)

db=Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#Pydantic model for structured output
class QueryVariations(BaseModel):
    queries:List[str]

original_query=""

llm_with_tools=llm.with_structured_output(QueryVariations)

prompt=f""" Generate 3 different variations of this query would help retrive relevant documents:
Original query:{original_query}

Return 3 alternate queries that rephrase or approach the same question from different angles.

"""

response=llm_with_tools.invoke(prompt)
query_variations=respone.queries

#Retrival system

retriver=db.as_retriever(search_kwargs={"k":3})
all_retrived_docs=[]

for i,query in enumerate(query_variations,1):
    print(f"Result for query {i}:{query}")

    docs=retriver.invoke(query)
    all_retrived_docs.append(docs) #Storing for RRF calculations

    print(f"Retrieved {len(docs)} documents")
    print("-"*50)

print("Multi-query retrival complete")


#output of all_rtetievd docs will have 3 lists with 3 lists  inside each

