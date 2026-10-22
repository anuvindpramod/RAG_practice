from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever 

chunks = [
    # Tesla - Financial & Production
    "Tesla reported record quarterly revenue of $25.2 billion in Q3 2024.",
    "Tesla's automotive gross margin improved to 19.3% this quarter.",
    "Tesla Cybertruck production ramp begins in 2024 with initial deliveries.",
    "Tesla announced plans to expand Gigafactory production capacity.",
    "Tesla stock price reached new highs following earnings announcement.",
    "Tesla's energy storage business grew 40% year-over-year.",
    "Tesla continues to lead in electric vehicle market share globally.",
    "Tesla Model Y became the best-selling vehicle worldwide.",
    "Tesla reported strong free cash flow generation of $7.5 billion.",
    "Tesla's Full Self-Driving revenue increased significantly.",
    
    # Microsoft - Development & Acquisitions
    "Microsoft acquired GitHub for $7.5 billion in 2018.",
    "Microsoft's cloud revenue Azure grew 29% year-over-year.",
    "Microsoft announced new AI features for Visual Studio Code.",
    "Microsoft Teams integration with GitHub enhances developer workflow.",
    "Microsoft's developer tools division sees strong adoption.",
    "Microsoft acquired Activision Blizzard for $68.7 billion.",
    "Microsoft's productivity suite gained 50 million new users.",
    "Microsoft announced new Surface devices for developers.",
    "Microsoft's AI Copilot features expand to more development tools.",
    "Microsoft's enterprise solutions drive revenue growth.",
    
    # NVIDIA - AI & Hardware
    "NVIDIA's data center revenue reached $47.5 billion annually.",
    "NVIDIA's H100 GPUs see unprecedented demand for AI training.",
    "NVIDIA announced next-generation Blackwell architecture.",
    "NVIDIA's gaming revenue declined due to crypto market changes.",
    "NVIDIA's automotive AI platform partnerships expanded.",
    "NVIDIA's AI chip shortage affects cloud providers.",
    "NVIDIA stock valuation exceeds $2 trillion market cap.",
    "NVIDIA's CUDA platform dominates AI development.",
    "NVIDIA announced new AI inference chips for edge computing.",
    "NVIDIA's partnership with major cloud providers strengthens.",
    
    # Google/Alphabet - AI & Cloud
    "Google's AI investments total over $100 billion in recent years.",
    "Google Cloud revenue grew 35% reaching $8.4 billion quarterly.",
    "Google announced Gemini AI model competing with GPT-4.",
    "Google's search advertising revenue remains strong at $59 billion.",
    "Google's Workspace products integrate advanced AI features.",
    "Google announced quantum computing breakthroughs.",
    "Google's autonomous vehicle division Waymo expands operations.",
    "Google's AI research published breakthrough papers.",
    "Google's cloud AI services see enterprise adoption.",
    "Google faces regulatory scrutiny over AI dominance.",
    
    # Noisy/Less Relevant Chunks
    "The Tesla coil was invented by Nikola Tesla in 1891.",
    "Microsoft Excel spreadsheet formulas can be complex for beginners.",
    "NVIDIA Shield TV streaming device gets software update.",
    "Google Maps navigation improved with real-time traffic data.",
    "Production delays affected multiple manufacturing sectors.",
    "Financial markets showed volatility during earnings season.",
    "Revenue recognition standards changed for software companies.",
    "Hardware components face supply chain constraints globally.",
    "Development tools market grows with remote work trends.",
    "AI research requires significant computational resources.",
    "Quarterly reports show mixed results across tech sector.",
    "Stock market analysts upgrade technology sector ratings.",
    "Cloud computing adoption accelerates in enterprise market.",
    "Data center construction increases globally.",
    "Semiconductor shortage impacts various industries.",
    "Electric vehicle charging infrastructure expands rapidly.",
    "Software development productivity tools gain popularity.",
    "Machine learning frameworks become more accessible.",
    "Enterprise software licensing models evolve.",
    "Technology conferences showcase latest innovations."
]

documents=[Document(page_content=chunk,
metadata={"source": f" chunk {i}"}
) for i,chunk in enumerate(chunks)
]
#Vector Retriver
embedding_model=OpenAIEmbeddings(model="text-embedding-3 small")
vector_store=Chroma.from_documents(
    embedding=embedding_model,
    documents=documents,
    collection_metadata={"hnsw:space":"cosine"}
)
vector_retriever=vector_store.as_retriever(search_kwargs={"k":15})

#bm25 retriever
bm25_retriver=BM25Retriever.from_documents(documents)
bm25_retriver.k=15

#Ensemble retriever
hybrid_search=EnsembleRetriever(
    retrievers=[vector_retriever,bm25_retriver],
    weights=[0.7,0.3]
)
 
query = "Tesla financial performance and production updates"

print("STEP 1: Hybrid Search Results")
print("-"*50)

retrieved_docs = hybrid_retriever.invoke(query)  # Get top 25 for reranking

#reranker with cohere
reranker=CohereRerank(model="rerank-english-v3.0",top_n=10)
reranked_docs=reranker.compress_documents(retrieved_docs,query)
top_reranked = reranked_docs[:5]

print("\n" + "="*80)
print("FINAL: RAG with Reranked Context")
print("-"*40)

llm=ChatOpenAI(model="gpt-4o")

prompt=f""" Based on the follwing documents, answer this question:{query}

Documents:
{chr(10).join(f"-{doc.page_content}" for doc in top_reranked)}

Please provide a clear, helpful answer using only the information from these documents.
"""

messages=[
    SystemMessage(content="You are a very helpful assistant"),
    HumanMessage(content=prompt)
]

result=model.invoke(messages)
print(result.content)