from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
import time

load_dotenv()

persist_directory = "db/chroma"

embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

retriver=db.as_retriever(search_kwargs={"k":5})

# retriever=db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k":5,
#         "search_score_threshold":0.3 #hyperparameter
#         }
# )

query="What is a very interesting fact about google?"

s_time=time.perf_counter()
relevant_docs=retriver.invoke(query)



print(f"User Query:{query} ")

print("------context retrieved ------")
for i, doc in enumerate(relevant_docs,1):
    print(f"Context {i}: \n{doc.page_content}\n")

combined_input=f"""Based on the follwing documents answer this question:{query}

Documents:

{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a detailed answer with references to the documents, if you are unable to find the answer in the documents, please respond with "I don't know" and do not make up an answer.

"""

model=ChatOpenAI(model="gpt-4o")

messages=[
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents. If the answer is not present in the documents, respond with 'I don't know' and do not fabricate information."),
    HumanMessage(content=combined_input)
]

result=model.invoke(messages)

e_time=time.perf_counter()
latency=e_time-s_time
print(f"latency: {latency}")
print("\n---Generated answer---")
# print("full result:",result)
print(result.content)
