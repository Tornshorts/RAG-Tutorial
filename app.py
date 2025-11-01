from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import shutil
from werkzeug.utils import secure_filename
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

app = Flask(__name__)
CORS(app)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

CHROMA_PATH = 'chroma'
DATA_PATH = "data"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def setup_documents():
    print("📚 Loading and processing documents...")
    
    # Load documents
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    
    if not documents:
        return {"error": "No documents found in data directory"}
    
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
        return {"message": f"Added {len(new_chunks)} new documents", "total_chunks": len(chunks_with_ids)}
    else:
        return {"message": "No new documents to add", "total_chunks": len(chunks_with_ids)}

def get_qa_chain():
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
    
    return qa

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_documents', methods=['POST'])
def load_documents():
    try:
        result = setup_documents()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        
        qa = get_qa_chain()
        result = qa.invoke(query)
        
        # Format sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "source": os.path.basename(doc.metadata.get('source', '')),
                "page": doc.metadata.get('page', 0)
            })
        
        return jsonify({
            "answer": result['result'],
            "sources": sources
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        # Ensure data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(DATA_PATH, filename)
                file.save(file_path)
                uploaded_files.append(filename)
            elif file.filename:
                return jsonify({"error": f"File {file.filename} is not a PDF"}), 400
        
        if not uploaded_files:
            return jsonify({"error": "No valid PDF files uploaded"}), 400
        
        return jsonify({
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    try:
        if not os.path.exists(DATA_PATH):
            return jsonify({"files": []})
        
        files = []
        for filename in os.listdir(DATA_PATH):
            if filename.endswith('.pdf'):
                file_path = os.path.join(DATA_PATH, filename)
                files.append({
                    "name": filename,
                    "size": os.path.getsize(file_path)
                })
        
        return jsonify({"files": files})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join(DATA_PATH, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        os.remove(file_path)
        return jsonify({"message": f"File {filename} deleted successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    try:
        if os.path.exists(CHROMA_PATH):
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function()
            )
            count = db._collection.count()
            return jsonify({"documents_loaded": count > 0, "total_chunks": count})
        else:
            return jsonify({"documents_loaded": False, "total_chunks": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)