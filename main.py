from multimodal_rag import partition_document, chunk_elements_by_title,summarise_chunks,create_vector_store,generate_final_answer
from time import perf_counter
from pathlib import Path
def main():
   
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
    
    query=Path(__file__).with_name("query.text").read_text(encoding="utf-8").strip()
    retriever=db.as_retriever(search_kwargs={"k":2})
    chunks=retriever.invoke(query)
    final_answer=generate_final_answer(chunks, query)
    print(final_answer)


if __name__ == "__main__":
    main()
