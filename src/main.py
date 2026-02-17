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

# Global variable for the RAG chain
qa_chain = None

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
    Endpoint to receive a PDF, save it, and build the RAG pipeline.
    """
    global qa_chain
    
    # Basic validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 1. Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📄 Saved {file.filename}. Initializing vector store...")
        
        # 2. Build the RAG chain using the newly uploaded file
        # Note: initialize_rag_pipeline now needs to accept the file path!
        qa_chain = initialize_rag_pipeline(file_path)
        
        return {"message": f"Successfully processed {file.filename}. You can now ask questions!"}
        
    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Failed to process PDF: {str(e)}"})


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=400, detail="Please upload a PDF document first before asking questions.")
    
    try:
        print(f"📝 Received Query: {request.query}")
        
        # Run the agent
        response = qa_chain.invoke({"input": request.query})
        answer = response['output']
        
        # --- NEW: Extract which tools the agent actually used! ---
        used_tools = []
        if "intermediate_steps" in response:
            for action, observation in response["intermediate_steps"]:
                # action.tool contains the name of the tool used (e.g., 'web_search')
                used_tools.append(action.tool) 
                
        # Remove duplicates, or provide a fallback if it answered from memory
        unique_sources = list(set(used_tools)) if used_tools else ["Agent Internal Knowledge"]
        
        return {"answer": answer, "sources": unique_sources}
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
    