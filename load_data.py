#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import argparse
from ast import main
import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function

CHROMA_PATH ='chroma'
DATA_PATH = "data"
#load documents
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

#Split documents
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =500,
        chunk_overlap=50,
        length_function = len, 
        is_separator_regex= False
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)
print (chunks)

#creating a database
def add_to_chroma (chuncks: list [Document]):
    db = Chroma( 
        persist_directory = CHROMA_PATH, 
        embedding_function = get_embedding_function()
    )
def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source} : {page}"

        #If the page id is same as last one increment last one 
        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index=0
        
        #calculate chunk id
        chunk_id =f"{current_page_id}:{current_chunk_index}"
        last_page_id=current_page_id

        chunk.metadata["id"]=chunk_id

    return chunks 
#Create the dataset object 
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function()
)
#Calculate Page Ids
chunks_with_ids = calculate_chunk_ids(chunks)

#Add or update the documents
existing_items = db.get(include=[])
existing_ids = set(existing_items["ids"])
print(f"👉 👉  Number of existing documents in DB:{len(existing_ids)}")

#only add documents that don't exist in db

new_chunkS=[chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]


if new_chunkS:
    print(f"👉 Add new documents:{len(new_chunkS)}")
    texts = [chunk.page_content for chunk in new_chunkS]
    metadatas = [chunk.metadata for chunk in new_chunkS]
    ids = [chunk.metadata["id"] for chunk in new_chunkS]
    db.add_texts(texts,metadatas=metadatas,ids=ids)
    db.persist()

else:
    print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source} : {page}"

        #If the page id is same as last one increment last one 
        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index=0
        
        #calculate chunk id
        chunk_id =f"{current_page_id}:{current_chunk_index}"
        last_page_id=current_page_id

        chunk.metadata["id"]=chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action = "store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    documents= load_documents()
    chunks=split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()