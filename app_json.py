from fastapi import FastAPI, HTTPException
import os
import time
import uvicorn
import json
import logging
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import JSONLoader
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Ollama endpoint
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11435"

# FastAPI instance
app = FastAPI(title="RAG Document Q&A API with HuggingFace + FAISS + Ollama")

# Globals
embeddings = None
vector_store = None
retrieval_chain = None

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the context provided.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Request and Response Schemas
class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    response_time: float
    context: List[Dict[str, Any]] = []

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, vector_store, retrieval_chain
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatOllama(model="mistral:7b-instruct-q4_0", base_url="http://127.0.0.1:11435")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logger.info("✅ FAISS index loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️ FAISS index not loaded. Run /create-embeddings first. Error: {e}")
    
    yield
    logger.info("🛑 Application shutting down.")

# Attach lifespan to FastAPI app
app.lifespan = lifespan

# Endpoint to create FAISS vector store
@app.post("/create-embeddings_json")
async def create_embeddings():
    global embeddings, vector_store, retrieval_chain
    try:
        json_dir = "data_base"
        processed_dir = "data_base_processed"
        input_file = "rag_data.json"  # Your input JSON file
        processed_file = "processed_hospital_data.json"

        # Create directories if they don't exist
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        input_path = os.path.join(json_dir, input_file)
        processed_path = os.path.join(processed_dir, processed_file)

        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file '{input_path}' not found.")
            raise HTTPException(status_code=404, detail=f"Input file '{input_path}' not found.")

        # Read and preprocess hospital data
        with open(input_path, "r") as f:
            hospital_data = json.load(f)

        processed_data = []
        for record in hospital_data:
            # Extract diagnosis and procedure descriptions
            diagnosis_desc = (
                record["diagnosis"][0]["description"]
                if record.get("diagnosis") and len(record["diagnosis"]) > 0
                else "No diagnosis"
            )
            procedure_desc = (
                record["procedures"][0]["description"]
                if record.get("procedures") and len(record["procedures"]) > 0
                else "No procedure"
            )
            # Create content field
            content = (
                f"{record['hospital_name']} - "
                f"Diagnosis: {diagnosis_desc} - "
                f"Procedure: {procedure_desc}"
            )
            processed_data.append({
                "content": content,
                "metadata": {
                    "hospital_id": record["hospital_id"],
                    "patient_id": record["patient_id"],
                    "location": record["location"],
                    "admission_date": record["admission_date"],
                    "outcome": record["outcome"]
                }
            })

        # Save processed data to a new JSON file
        with open(processed_path, "w") as f:
            json.dump(processed_data, f, indent=2)
        logger.info(f"Processed data saved to '{processed_path}'.")

        # Load processed JSON data
        loader = JSONLoader(
            file_path=processed_path,
            jq_schema=".[] | .content",
            text_content=True
        )
        docs = loader.load()

        if not docs:
            logger.error("No documents loaded from processed JSON.")
            raise HTTPException(status_code=404, detail="No documents loaded from processed JSON.")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        logger.info(f"Loaded and split {len(final_documents)} documents.")

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(final_documents, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("FAISS vector store created and saved.")

        # Initialize LLM and retrieval chain
        llm = ChatOllama(model="mistral:7b-instruct-q4_0", base_url="http://127.0.0.1:11435")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return {
            "status": "success",
            "message": "FAISS vector store created successfully.",
            "document_count": len(final_documents)
        }
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    global retrieval_chain
    if not retrieval_chain:
        logger.error("Vector store not initialized.")
        raise HTTPException(
            status_code=400,
            detail="Vector store not initialized. Call /create-embeddings first."
        )
    try:
        start = time.process_time()
        response = retrieval_chain.invoke({"input": request.query})
        duration = time.process_time() - start
        context_docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in response["context"]
        ]
        logger.info(f"Query processed in {duration:.2f} seconds.")
        return {
            "answer": response["answer"],
            "response_time": duration,
            "context": context_docs
        }
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store_loaded": retrieval_chain is not None
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("app_json:app", host="0.0.0.0", port=8000, reload=True)