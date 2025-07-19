import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function


def setup_documents():
    CHROMA_PATH = 'chroma'
    DATA_PATH = "data"
    
    print("📚 Loading and processing documents...")
    
    # Load documents
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    
    # Calculate chunk IDs
    def calculate_chunk_ids(chunks):
        last_page_id = None
        current_chunk_index = 0
        
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id
        
        return chunks
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Setup database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    # Add new documents
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"➕ Adding {len(new_chunks)} new documents...")
        texts = [chunk.page_content for chunk in new_chunks]
        metadatas = [chunk.metadata for chunk in new_chunks]
        ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_texts(texts, metadatas=metadatas, ids=ids)
    else:
        print("✅ No new documents to add")
    
    return db


def start_qa_session():
    CHROMA_PATH = "chroma"
    
    print("🤖 Starting Q&A session...")
    
    # Initialize the retriever
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    retriever = db.as_retriever()
    
    # Set up the LLM
    llm = OllamaLLM(model="llama3.2:latest")
    
    # Build the QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Ask questions
    print("🎯 Ready! Ask your questions (type 'exit' to quit):\n")
    
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
            
        if query.strip():
            try:
                result = qa.invoke(query)
                print(f"\n🧠 Answer: {result['result']}\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

# Main execution
if __name__ == "__main__":
    print("🚀 Starting RAG System...\n")
    
    # Step 1: Setup documents
    setup_documents()
    
    # Step 2: Start Q&A
    start_qa_session()