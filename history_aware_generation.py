from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import time

load_dotenv()

persist_directory = "db/chroma"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    start_time = time.perf_counter()
    print(f"\n-- You asked: {user_question}")

    if chat_history:
        messages = [
            SystemMessage(
                content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
            ),
            *chat_history,
            HumanMessage(content=f"New question: {user_question}"),
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for {search_question} in the documents...")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    combined_input = f"""Based on the following documents, answer this question: {user_question}

    Documents:

    {chr(10).join(f"- {doc.page_content}" for doc in docs)}

    Please provide a detailed answer with references to the documents, if you are unable to find the answer in the documents, please respond with "I don't know" and do not make up an answer.

    """

    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided documents. If the answer is not present in the documents, respond with 'I don't know' and do not fabricate information."
        ),
        *chat_history,
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    print(f"Latency: {time.perf_counter() - start_time:.2f} seconds")

    return answer


def start_chat():
    print("Welcome to the chat! Type 'exit' to end the conversation.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        answer = ask_question(question)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    start_chat()
