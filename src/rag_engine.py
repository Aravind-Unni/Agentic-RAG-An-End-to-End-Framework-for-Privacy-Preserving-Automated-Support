import os
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# LangGraph Imports
from langgraph.graph import StateGraph, START, END

load_dotenv()

COLLECTION_NAME = "dynamic_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# 1. Define the State & Schemas
# ==========================================
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_required: bool
    retrieval_attempts: int

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ..., description="Documents are relevant to the question, 'yes' or 'no'"
    )

def initialize_rag_pipeline(file_path):
    print(f"📄 Loading PDF from {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print(f"✂️ STEP 2: TEXT CHUNKING")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_chunks = text_splitter.split_documents(docs)
    
    print("🧠 Initializing local HuggingFace embedding model...")
    # Using local CPU to avoid HuggingFace API cold starts and batch size limits
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    print("💾 Creating new Chroma vector store in memory...")
    vector_store = Chroma.from_documents(
        documents=all_chunks, embedding=embedding_model, collection_name=COLLECTION_NAME 
    )
    # Retrieving 6 chunks to give the Grader LLM plenty of context to evaluate
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("🤖 Initializing Groq Agent Fleet...")
    # Agent 1: Grader (Heavy logic to evaluate context accurately)
    grader_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # Agent 2: Rewriter (Creative semantic understanding for rephrasing)
    rewriter_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    # Agent 3: Generator (Heavy synthesizer for writing the final answer)
    generator_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    print("🤖 Initializing Groq Agent Fleet...")
    

    
    print("🔌 Connecting Tavily API Key explicitly...")
    tavily_key = os.environ.get("TAVILY_API_KEY") 
    tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
    tavily_tool = TavilySearchResults(api_wrapper=tavily_wrapper, max_results=3)
    

    # ==========================================
    # 2. Define the Nodes (The Workers)
    # ==========================================
    def retrieve(state: GraphState):
        """Tool: Fetch from local ChromaDB"""
        print("---RETRIEVE FROM CHROMA---")
        question = state["question"]
        attempts = state.get("retrieval_attempts", 0) + 1
        documents = retriever.invoke(question)
        return {"documents": documents, "retrieval_attempts": attempts}

    def web_search(state: GraphState):
        """Tool: Fetch from Tavily (Fallback)"""
        print("---WEB SEARCH FALLBACK---")
        question = state["question"]
        
        try:
            # 1. Get the raw list of dictionaries from Tavily
            raw_docs = tavily_tool.invoke({"query": question})
            
            # 2. Extract just the 'content' from each dictionary and join with double newlines
            web_text = "\n\n".join([d.get("content", "") for d in raw_docs if isinstance(d, dict)])
            
            # Fallback just in case Tavily returns an unexpected format
            if not web_text:
                web_text = str(raw_docs)
            
            # 3. Create the Document using the joined string
            web_results_doc = Document(page_content=web_text)
            
            return {"documents": [web_results_doc]}
            
        except Exception as e:
            print(f"❌ Web Search Error parsing results: {e}")
            # Prevent the graph from crashing by passing a dummy document
            error_doc = Document(page_content="I attempted a web search but encountered an error parsing the data.")
            return {"documents": [error_doc]}

    def grade_documents(state: GraphState):
        """Agent: Filter irrelevant chunks in a SINGLE call to avoid rate limits"""
        print("---GRADE DOCUMENTS (BULK)---")
        question = state["question"]
        documents = state["documents"]
        
        context = "\n\n".join([f"Doc {i}: {d.page_content}" for i, d in enumerate(documents)])

        
        system = """You are a strict grader assessing relevance of retrieved documents to a user question. 
        Look at the documents carefully. If they do NOT contain the actual answer or facts to address the question, respond with 'no'.
        Respond with ONLY the word 'yes' if the answer is explicitly in the text, or 'no' if it is not."""
        
        response = grader_llm.invoke(f"{system}\n\nQuestion: {question}\n\nContext: {context}")
        decision = response.content.strip().lower()


        web_search_required = True if "no" in decision else False
        
        # If relevant, keep documents; if not, clear them to trigger rewrite
        filtered_docs = documents if not web_search_required else []
                
        return {"documents": filtered_docs, "web_search_required": web_search_required}

    def rewrite_query(state: GraphState):
        """Agent: Formulate a better question"""
        print("---REWRITE QUERY---")
        question = state["question"]
        
        # --- THE FIX: Make the prompt extremely strict ---
        system = """You are an expert question re-writer. Your job is to convert an input question into a better, highly targeted search query.
        Look at the input and identify the core entity and intent.
        CRITICAL RULE: You must respond with ONLY the new question. Do not include any preambles, conversational filler, or explanations."""
        
        rewrite_prompt = PromptTemplate.from_template(f"{system}\n\nInitial question: {question}\nImproved question:")
        
        chain = rewrite_prompt | rewriter_llm
        better_question = chain.invoke({"question": question}).content.strip() # Strip removes accidental white spaces
        
        # Print this so you can see exactly what it's sending to Tavily!
        print(f"🔄 Rewritten Query: {better_question}") 
        
        return {"question": better_question}

    def generate(state: GraphState):
        """Agent: Write the final answer"""
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        
        system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
        prompt = PromptTemplate.from_template(f"{system}\n\nQuestion: {{question}} \nContext: {{context}} \nAnswer:")
        
        docs_text = "\n\n".join([d.page_content for d in documents])
        chain = prompt | generator_llm
        generation = chain.invoke({"context": docs_text, "question": question}).content
        return {"generation": generation}

    # ==========================================
    # 3. Define the Routing Logic (The Edges)
    # ==========================================
    def decide_to_generate(state: GraphState):
        """Edge: Implement Max 5 Retry Logic with Web Fallback"""
        web_search_required = state["web_search_required"]
        attempts = state.get("retrieval_attempts", 0)
        
        if web_search_required:
            print(f"---EVALUATION: Docs irrelevant. Attempt {attempts} of 5---")
            if attempts >= 2:
                print("---MAX ATTEMPTS REACHED: FALLING BACK TO WEB SEARCH---")
                return "web_search"
            else:
                print("---LOOPING BACK TO REWRITE---")
                return "rewrite_query"
        else:
            print("---EVALUATION: Docs relevant. Proceeding to generation---")
            return "generate"

    # ==========================================
    # 4. Compile the Graph
    # ==========================================
    print("🚀 Compiling LangGraph Workflow...")
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)

    # Add Edges (Notice the direct flow from START to retrieve)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate)
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    print("✅ LangGraph Ready.")
    return app