import uvicorn
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_engine import initialize_rag_pipeline

# Initialize App
app = FastAPI()

# Mount Static Folder (This serves CSS/JS/Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable for the RAG chain
qa_chain = None

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    global qa_chain
    # Initialize the RAG logic once on startup
    qa_chain = initialize_rag_pipeline()

# --- Routes ---

@app.get("/")
async def index():
    # This serves your Chatbot Interface
    return FileResponse('static/index.html')

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    """
    This is the Chatbot API endpoint.
    It takes a JSON query, runs it through RAG, and returns the answer.
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG system is not initialized")
    
    try:
        print(f"📝 Received Query: {request.query}")
        
        # Run the query through your RAG pipeline
        response = qa_chain.invoke({"query": request.query})
        
        # Extract answer and sources
        answer = response['result']
        sources = [doc.metadata.get('source', 'Unknown') for doc in response.get('source_documents', [])]
        unique_sources = list(set(sources))
        
        return {"answer": answer, "sources": unique_sources}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    # This matches the style of your reference code
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
    