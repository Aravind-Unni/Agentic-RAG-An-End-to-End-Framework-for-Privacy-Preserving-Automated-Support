import uvicorn
import os
import shutil
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_engine import initialize_rag_pipeline

# Initialize App
app = FastAPI()

# Mount Static Folder (This serves CSS/JS/Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a directory to temporarily store uploaded PDFs
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variable for the compiled LangGraph application
rag_app = None

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str

# --- Routes ---

@app.get("/")
async def index():
    # This serves your Chatbot Interface
    return FileResponse('static/index.html')

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to receive a PDF, save it, and build the LangGraph RAG pipeline.
    """
    global rag_app
    
    # Basic validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 1. Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📄 Saved {file.filename}. Initializing LangGraph workflow...")
        
        # 2. Build the LangGraph app using the newly uploaded file
        rag_app = initialize_rag_pipeline(file_path)
        
        return {"message": f"Successfully processed {file.filename}. You can now ask questions!"}
        
    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Failed to process PDF: {str(e)}"})

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    global rag_app
    if not rag_app:
        raise HTTPException(status_code=400, detail="Please upload a PDF document first before asking questions.")
    
    try:
        print(f"📝 Received Query: {request.query}")
        
        # --- THE FIX: LangGraph State Dictionary requires the "question" key ---
        initial_state = {"question": request.query}
        
        # Run the LangGraph application. It returns the final state snapshot.
        final_state = rag_app.invoke(initial_state)
        
        # Extract the final answer from the updated state
        answer = final_state.get("generation", "I couldn't generate an answer based on the current context.")
        
        # --- NEW: Extracting sources based on LangGraph State ---
        used_tools = []
        # If the state flag for web search was flipped to True, it used Tavily
        if final_state.get("web_search_required"):
            used_tools.append("Tavily Web Search")
        # Otherwise, if there are documents in the state, it used the local PDF
        elif final_state.get("documents"):
            used_tools.append("Local PDF Document")
        else:
            used_tools.append("Agent Internal Knowledge")
            
        return {"answer": answer, "sources": used_tools}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
    