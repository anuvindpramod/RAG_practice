import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time



load_dotenv()

def document_loader(docs_path="rag_tut/docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified path '{docs_path}' does not exist.")

    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )    
    documents=loader.load()

    return documents

def split_documents(documents, chunk_size=800,chunk_overlap=0):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks,persist_directory="db/chroma"):
    embedding_model=OpenAIEmbeddings(
        model="text-embedding-3-small")
    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )    
    return vector_store