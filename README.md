# PolicyGuard AI: Agentic RAG for Document Q&A 🛡️

**PolicyGuard AI** is a privacy-first **Retrieval-Augmented Generation (RAG)** system designed to answer user questions based on specific policy documents (e.g., Return & Refund Policies).

It runs **entirely locally** using **Llama 3.2**, ensuring that sensitive data never leaves your machine. The system includes a robust evaluation pipeline to measure accuracy, hallucination avoidance, and answer clarity.

---

## 🚀 Key Features

* **Dynamic Document Upload:** Users can upload any PDF on the fly via the web interface.
* **In-Memory Vector Processing:** Uses **ChromaDB** to build ephemeral, in-memory vector stores, preventing file locks and ensuring clean states between uploads.
* **Agentic Reasoning (ReAct):** The LLM "thinks" before acting, deciding whether to search the local document or query the live internet based on the user's prompt.
* **Multi-Tool Integration:** * `search_policy_document`: A local retriever tool leveraging huggingface `sentence-transformers`.
  * `web_search`: A live internet search tool powered by the **Tavily API**.
* **Secure API Management:** Uses `python-dotenv` to keep Hugging Face and Tavily keys safe and out of version control.
* **Evaluation Suite:** Includes an "LLM-as-a-Judge" module to grade answers on accuracy, hallucination avoidance, and clarity automatically.

---

## 🏗️ Architecture Overview

The pipeline follows an advanced ReAct Agent workflow optimized for local execution:

1.  **Dynamic Ingestion:** A PDF is uploaded via the frontend and loaded using `PyPDFLoader`.
2.  **Chunking:** Text is split into chunks of 800 characters with a 150-character overlap using `RecursiveCharacterTextSplitter`.
3.  **Embedding & Storage:** Chunks are converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stored in an **In-Memory ChromaDB**.
4.  **Agentic Routing:** When a user asks a question, the **Llama 3.3** agent analyzes the request.
5.  **Tool Execution:** The agent autonomously selects a tool:
    * *Is the answer in the policy?* -> Action: `search_policy_document`
    * *Is it a real-world fact or current event?* -> Action: `web_search`
6.  **Observation & Generation:** The agent synthesizes the observations from its tools and generates a final formatted answer.

## 🛠️ Setup Instructions

### 1. Prerequisites
* **Python 3.10+**
* **Ollama:** Download and install from [ollama.com](https://ollama.com/).
* **API Keys:** You will need free API keys from [Hugging Face](https://huggingface.co/) and [Tavily](https://tavily.com/).
### 2. Install Dependencies
Navigate to the `src` folder and install the required Python libraries:

```bash
cd src
pip install -r requirements.txt
ollama pull llama3.2

```
## 📝 Prompt Engineering

You are a specific Policy Assistant for Rainbow Bazaar.
Your goal is to answer the user question based ONLY on the provided context chunks.

CONTEXT:
{context}

USER QUESTION:
{question}

---
STRICT RULES:
1. **Focus:** Answer the question directly. Do not comment on the quality or repetition of the context text.
2. **Grounding:** If the answer is found in *any* part of the context, use it. Ignore duplicates.
3. **No Filler:** Do not start with "I apologize" or "The context mentions." Start directly with the answer.
4. **Citation:** Support your answer with source IDs.
5. **Missing Info:** If the answer is strictly NOT in the context, say: "I cannot find this information in the policy."

FORMAT:
**Answer:** [Direct Answer]
**Details:** [Bullet points with citations]


## 📊 Evaluation Results
* ** check the RAGpipeline.ipynb file (simple RAG)

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | **88.9%** | Percentage of questions answered correctly according to the rubric. |
| **Hallucination Avoidance** | **66.7%** | Success rate in correctly refusing to answer out-of-scope questions (e.g., "Who is the CEO?"). |
| **Answer Clarity** | **75.6%** | Average score (3.8/5.0) for coherence and formatting. |

## ⚖️ Key Trade-offs & Improvements

### Trade-offs Taken
* **In-Memory vs Persistent DB:** We traded persistent storage for in-memory processing to allow dynamic PDF uploads without Windows file-locking conflicts. This requires re-embedding the document on every new upload.
* **Agentic Overhead:** Using a ReAct agent increases the time-to-first-token compared to a standard RAG chain, as the model must generate internal "Thought" and "Action" strings before finalizing the answer.
* **Model Size (Llama 3.2):** We opted for the smaller 3B parameter model for speed and low memory usage. While highly capable, smaller models require strict `handle_parsing_errors=True` parameters in the Agent Executor to recover if they deviate from the ReAct formatting.
* **Chunk Size (800 chars):** Chosen to balance context with retrieval speed. Smaller chunks might miss context, while larger ones introduce noise.

---

## 🌐 Deployment & Web Interface

PolicyGuard AI is fully deployable as a web application, featuring a high-performance **FastAPI backend** and a responsive **HTML/JS frontend**.

### **Backend (FastAPI)**
The core RAG logic has been wrapped in a RESTful API using **FastAPI**. This allows for asynchronous query processing and easy integration with other services.
* **Endpoints:** Supports `/api/upload` for dynamic document ingestion and `/api/ask` for agentic query processing.
* **Server:** Powered by **Uvicorn**, a lightning-fast ASGI server.

### **Frontend (Chat UI)**
Integrates a custom chat interface using the **HTML5 UP "Dimension" template**.
* **Features:** Dynamic file upload buttons, status indicators, starfield background animation, responsive modal window, and real-time chat masking until a document is successfully processed into the vector database.
* **Tech:** Pure HTML/CSS/JavaScript (No heavy frontend frameworks like React or Angular required).

