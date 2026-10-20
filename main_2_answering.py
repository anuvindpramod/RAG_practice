from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from multimodal_rag import generate_final_answer
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path

persistant_directory='db2/chroma_db'
embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")


db=Chroma(
    embedding_function=embedding_model,
    persist_directory=persistant_directory,
    collection_metadata={"hnsw:space":"cosine"}
)

def main():
    query=Path(__file__).with_name("query.text").read_text(encoding="utf-8").strip()
    retriever=db.as_retriever(search_kwargs={"k":2})
    chunks=retriever.invoke(query)
    final_answer=generate_final_answer(chunks,query)
    print(final_answer)

if __name__=="__main__":
    main()
