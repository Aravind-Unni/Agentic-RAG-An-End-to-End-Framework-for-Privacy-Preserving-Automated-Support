import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
# Keeping this import as per your request
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

def initialize_rag_pipeline():
    """
    Loads the existing Vector DB and initializes the LLM Chain.
    This runs ONCE when the server starts.
    """
    print("⏳ Loading Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"⏳ Loading Vector Store from {PERSIST_DIRECTORY}...")
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Chroma DB not found at {PERSIST_DIRECTORY}. Please run the ingestion notebook first.")

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_name="company_policies"
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print(f"⏳ Connecting to Ollama ({LLM_MODEL_NAME})...")
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0.1,
        num_predict=512
    )

    # Define the Prompt
    prompt_template = """You are a Policy Assistant. Answer based ONLY on the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # --- FIX IS HERE ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # <--- Added this so sources are returned
    )
    
    print("✅ RAG Pipeline Ready.")
    return qa_chain

if __name__ == "__main__":
    print("\n🔬 STARTING TEST MODE...")
    try:
        # 1. Initialize the chain
        chain = initialize_rag_pipeline()
        
        # 2. Ask a test question
        test_question = "What is the return policy?"
        print(f"\n❓ Asking: {test_question}")
        
        # 3. Invoke the chain
        response = chain.invoke({"query": test_question})
        
        print("\n✅ ANSWER GENERATED:")
        print("-" * 40)
        print(response['result'])
        print("-" * 40)
        
        print("\n📄 SOURCES:")
        # This will now work because we added return_source_documents=True
        for doc in response.get('source_documents', []):
            print(f"- {doc.metadata.get('source', 'Unknown')}")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")