import json
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from time import perf_counter


load_dotenv()

def partition_document(file_path):
    elements=partition_pdf(
        filename=file_path,
        strategy="hi_res",#fast strategy for partitioning the document
        infer_table_structure=True,#keeping tables as strcuted HTML
        extract_image_block_types=["Image"],# Grabing images
        extract_image_block_to_payload=True #for converting the images to base64 and storing in payload 
    )

    print(f"Exatraction of the following nuber of elements from the document: {len(elements)} from the document {os.path.basename(file_path)}")

    return elements

def chunk_elements_by_title(elements):
    """
    Create intelligent chunks using the title of the document

    """
    chunks=chunk_by_title(elements,
    max_characters=3000,
    new_after_n_chars=2400,
    combine_text_under_n_chars=400
    )
    print(f"Number of chunks created from the document: {len(chunks)}")
    return chunks

def seperate_content_types(chunk):
    """
    analysing the types of content in a chunk
    """
    content_data={
        'text': chunk.text,
        'tables':[],
        'images':[],
        'types':['text']
    }

    if hasattr(chunk,'metadata') and hasattr(chunk.metadata,'orig_elements'):
        for element in chunk.metadata.orig_elements or []:
            element_type=type(element).__name__

            if element_type=='Table':
                content_data['types'].append('table')
                table_html=getattr(element.metadata,'text_as_html',element.text)
                content_data['tables'].append(table_html)

            elif element_type=='Image':
                if hasattr(element,'metadata') and hasattr(element.metadata,'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types']=list(set(content_data['types']))
    
    return content_data


def summarise_chunks(chunks):
    print("Processing the chunks with AI summaries")
    langchain_documents=[]
    total_chunks=len(chunks)

    for i,chunk in enumerate(chunks,1):
        print(f"processing the chunk{i}/{total_chunks}")

        content_data=seperate_content_types(chunk)
        print(f"The types of data in the chunk is {content_data['types']}")
        print(f"Tables{len(content_data['tables'])}, Images {len(content_data['images'])}")

        #create ai enhanced summary of the chunks with tables and images

        if content_data['tables'] or content_data['images']:
            print("Creating ai summry of mixed content")
            try:
                enhanced_content=create_ai_summary(
                    content_data['tables'],
                    content_data['images'],
                    content_data['text']
                )
                print("Sucess in creating the summary")
            except Exception as e:
                print(f"failed due to {e}")
                enhanced_content=content_data['text']
        else:
            enhanced_content=content_data['text']

        doc=Document(
            page_content=enhanced_content,
            metadata={
                "original_content":json.dumps(
                    {
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "image_base64":content_data['images']
                    }
                )
            }
        ) 
        langchain_documents.append(doc)

    print("Proceing into langchain documents")
    return langchain_documents



def create_ai_summary(tables:list[str],images:list[str],text: str):
    try:
        llm=ChatOpenAI(model='gpt-4o',temperature=0)
        prompt_text=f""" You are creating a searchable description for document content retrieval.
        CONTENT TO ANALYSE:
        TEXT CONTENT:
        {text}

        """

        if tables:
            prompt_text+="TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text+=f"Table {i+1}:\n {table}\n\n"
        prompt_text+="""
                YOUR TASK:
                Generate a comprehensive searchable description that covers:

                1. Key facts, numbers, and data poijts from text and tables
                2. Main topics and concepts dicussed
                3. Questions this content could answer
                4. Visual content analysis (charts, diagram,patterns in images)
                5. Alternative search terms users might use
               
               MAke it detailed and searchable- prioritize findability over brevity
                
               SEARCHBLE DESCRIPTION: 
                """

        message_content=[{"type":"text","text":prompt_text}]

        for image_base64 in images:
            message_content.append(
                {"type":"image_url",
                "image_url":{"url":f"data:image/jpeg;base64,{image_base64}"}
                }
            )

        message=HumanMessage(content=message_content)
        response=llm.invoke([message])

        return response.content  

    except Exception as e:
        print(f"Failed due o {e}")
        summary=f"{text[:300]}..."
        return summary
    
def create_vector_store(documents, persist_directory="db2/chroma_db"):
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store=Chroma.from_documents(
        persist_directory=persist_directory,
        embedding=embedding_model,
        documents=documents,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("Vectorstore saved sucessfully")

    return vector_store

def generate_final_answer(chunks, query):
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_text = f"""
        Based on the following documents, please answer this query: {query}

        CONTENT TO ANALYZE:
        """
        images = []
        for i, chunk in enumerate(chunks):
            prompt_text += f"-- Document {i + 1} ---\n"
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                prompt_text += original_data.get("raw_text", chunk.page_content) + "\n"
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j + 1}:\n{table}\n\n"
                images.extend(original_data.get("image_base64", []))
            else:
                prompt_text += chunk.page_content + "\n"
            prompt_text += "\n"

        prompt_text += """
        Please provide a clear, comprehensive answer using the text, tables, and images above.
        If the documents don't contain sufficient information to answer the question, say
        "I don't have enough information to answer that question based on the provided documents."

        ANSWER:"""
        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Failed to generate the answer: {e}")
        return "Sorry, I encountered an error while generating the answer"


if __name__ == "__main__":
    file_path="/Users/anuvindpramod/Desktop/rag_tut/atttention.pdf"
    start_time=perf_counter()
    elements=partition_document(file_path)
    end_time=perf_counter()
    print(f"Time taken to partition the document: {end_time - start_time} seconds")
    #element[32].to_dict() #to see the structure of the element
    # for element in elements:
    #     table_count=0
   
    #     if element.category=="Table":
    #         table_count+=1
    #         print(f"Table {table_count} found in the document with the following structure: {element.to_dict()}")
    
    chunks=chunk_elements_by_title(elements)
    #chunks[0].to_dict() #to see the structure of the chunk
    #chunks[3].metadata.orig_elements[3].to_dict() #to see the structure of the element in the chunk
    processed_chunks=summarise_chunks(chunks) 
    db=create_vector_store(processed_chunks)

    


    

