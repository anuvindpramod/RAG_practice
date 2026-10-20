from dotenv import load_dotenv  
from langchain_openai import ChatOpenAI
from ingestion_pipeline import document_loader

load_dotenv()

llm=ChatOpenAI(model="gpt-4o",temperature=0.0)

document=document_loader(docs_path="/Users/anuvindpramod/Desktop/rag_tut/docs")
doc=document[0]

prompt=f"""
You are a text chunking expert. Split the inputted text into logical chunks.

Rules:
Each chunk should be a logical unit of information and should keep related information together.
Each chunk should be 200 characters or less
Insert "<<<SPLIT>>>" between each chunk

Text {doc.page_content}

Return the text with the inserted "<<<SPLIT>>>" markers between each chunk. 

"""
print("Respone from AI")
respone=llm.invoke(prompt)
output_text=respone.content

chunks=output_text.split("<<<SPLIT>>>")

cleaned_up=[]
for chunk in chunks:
    cleaned_chunk=chunk.strip()
    if cleaned_chunk:
        cleaned_up.append(cleaned_chunk)


for i, chunk in enumerate(cleaned_up,1):
    print(f"Chunk{i} with char {len(chunk)}")
    print(f"{chunk}")

