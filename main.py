import os
import ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def load_and_split(docs_dir="docs"):
    documents = []
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(filepath).load())
        elif filename.endswith(".txt"):
            documents.extend(TextLoader(filepath, encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return splitter.split_documents(documents)

def get_vectorstore(chunks, persist_dir="./chroma_db"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(persist_dir):
        print("Загружаю существующую базу...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("Создаю новую базу...")
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

def rag_query(query, vectorstore, model="llama3.2", k=3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Отвечай строго по контексту. Если ответа нет — скажи об этом.
Контекст:
{context}
Вопрос: {query}
Ответ:"""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    answer = ""
    for chunk in stream:
        piece = chunk["message"]["content"]
        answer += piece
        print(piece, end="", flush=True)
    print()
    return answer

if __name__ == "__main__":
    chunks = load_and_split("docs")
    print(f"Фрагментов: {len(chunks)}")
    vectorstore = get_vectorstore(chunks)
    while True:
        q = input("\nВы: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Ассистент: ", end="", flush=True)
        rag_query(q, vectorstore)