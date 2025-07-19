from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM  
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

# Initialize the retriever
CHROMA_PATH = "chroma"
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function()
)

retriever = db.as_retriever()

# Set up the LLM with updated OllamaLLM
llm = OllamaLLM(model="llama3.2:latest") #You can change to your prefered model

# Build the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question loop
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa.invoke(query)  
    print("\n🧠 Answer:", result["result"])

    
