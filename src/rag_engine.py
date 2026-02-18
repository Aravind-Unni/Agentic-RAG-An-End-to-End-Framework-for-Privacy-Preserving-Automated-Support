import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# REMOVED: from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq # ADDED
from dotenv import load_dotenv
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()


COLLECTION_NAME = "dynamic_policies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# CHANGED: Update to Groq's Llama 3.3 70B model
LLM_MODEL_NAME = "llama-3.3-70b-versatile" 

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def initialize_rag_pipeline(file_path):
    print(f"📄 Loading PDF from {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print(f"✂️ STEP 2: TEXT CHUNKING (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_chunks = text_splitter.split_documents(docs)
    
    print("🧠 Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    print("💾 Creating new Chroma vector store in memory...")
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME 
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # CHANGED: Initialize ChatGroq instead of ChatOllama
    print(f"🤖 Connecting to Groq ({LLM_MODEL_NAME})...")
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0.2, max_retries=2)

    print("🛠️ Setting up Agent Tools...")
    
    # Tool 1: The Local PDF Retriever
    policy_tool = create_retriever_tool(
        retriever,
        name="search_policy_document",
        description="Searches the uploaded document. Use this FIRST to find company rules, refund policies, numbers, or facts."
    )
    
    # Tool 2: Tavily Web Search
    tavily_tool = TavilySearchResults(
        max_results=3,
        name="web_search",
        description="Use this tool to search the internet for current events, real-world facts, or information NOT found in the policy document."
    )
    
    # Give the agent access to both tools!
    tools = [policy_tool, tavily_tool]

    react_prompt = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(react_prompt)

    print("🧠 Initializing ReAct Agent...")
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    print("🚀 Multi-Tool Agentic RAG Ready.")
    return agent_executor