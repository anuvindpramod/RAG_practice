from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from ingestion_pipeline import document_loader
import time

load_dotenv()

time_s=time.perf_counter()
document=document_loader(docs_path="/Users/anuvindpramod/Desktop/rag_tut/docs")
doc=document[0]


semantic_splitter=SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=0.70
)

chunks=semantic_splitter.split_text(doc.page_content)

time_e=time.perf_counter()
latency=time_e-time_s
print("---- results ----")
for i,chunk in enumerate(chunks,1):
    print(f" chunk {i} with character={len(chunk)}")
    print(f"{chunk}")
    print(latency)
