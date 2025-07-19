from langchain_ollama import OllamaEmbeddings

#Embedding function

def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
       # base_url="" 
    )
    return embeddings