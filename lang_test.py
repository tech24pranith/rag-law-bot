import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, List, Dict
import numpy as np
import datetime
import pandas as pd
import csv 
import warnings
from langdetect import detect, LangDetectException
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage 

# --- External RAG Dependencies ---
from sentence_transformers import SentenceTransformer
import chromadb

# Suppress warnings from Sentence Transformers/ChromaDB initialization
warnings.filterwarnings("ignore")

# Define global constants
LOG_FILE_PATH = "user_logs.csv"
ADMIN_PASSWORD = 'admin' # Global password

# Load environment variables (assumes GEMINI_API_KEY is in a .env file)
load_dotenv()

# LOGGING AND CLEANUP FUNCTIONS
def check_and_reset_log(file_path):
    """Deletes the log file if a corruption marker is present in st.session_state."""
    if 'log_reset_needed' in st.session_state and st.session_state.log_reset_needed:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                st.session_state.log_reset_needed = False
                st.toast("✅ Corrupted log file reset successfully!", icon="🧹")
            except Exception as e:
                st.error(f"Failed to delete corrupted log file: {e}")

def log_user_interaction(query: str, response: str, retriever_count: int):
    """Appends interaction details to a CSV log file using the standard csv module for reliability."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = [
        timestamp,
        query,
        response,
        retriever_count
    ]
    
    file_exists = os.path.exists(LOG_FILE_PATH)
    
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8', newline="") as f:
            writer = csv.writer(f) 
            
            if not file_exists or os.stat(LOG_FILE_PATH).st_size == 0:
                writer.writerow(["Timestamp", "Query", "Response", "Retrieved_Chunks"])
            
            writer.writerow(log_entry)
            
    except Exception:
        pass 

# 2. RAG Component Classes 
class EmbeddingManager:
    """Handles document embedding generation using Sentence Transformer"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            st.error(f"Error loading embedding model {self.model_name}: {e}")
            
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded.")
        embeddings = self.model.encode(texts, show_progress_bar=False) 
        return embeddings

class VectorStore:
    """Manages document embeddings in a chromadb vector store"""
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        current_dir = Path(__file__).parent
        self.persist_directory = str(current_dir / persist_directory)

        self.client = None
        self.collection = None
        self.collection_count = 0
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True) 
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"desciption": "PDF document embedddings for RAG"}
            )
            self.collection_count = self.collection.count()
        except Exception as e:
            st.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        pass

class RAGRetriever:
    """ Handles query-based retrieval from the vector store """
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        if self.vector_store.collection.count() == 0:
            return []
            
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
            return retrieved_docs
            
        except Exception as e:
            st.error(f"Error during retrieval: {e}") 
            return []

def rag_simple(query: str, retriever: RAGRetriever, llm: ChatGoogleGenerativeAI, top_k: int = 3, history: List[Dict[str, str]] = None) -> str:
    """Performs RAG with In-Context Language Handling via the Gemini LLM."""
    original_lang_code = 'en'
    try:
        original_lang_code = detect(query)
    except LangDetectException:
        original_lang_code = 'en'
    
    # Retrieve context based on the original query (relying on multilingual embeddings)
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    st.session_state.retrieved_count = len(results)

    # 2. Format the Conversation History
    chat_history_messages = []
    if history:
        for message in history:
            if message['role'] == 'user':
                chat_history_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                chat_history_messages.append(AIMessage(content=message['content']))
    
    # 3. Define the System Prompt (Instructing Translation)
    # The prompt tells Gemini to handle the internal translation (Query -> English RAG -> Answer -> User Language)
    system_instruction = f"""You are a legal helper in India with high knowledge in your field, specializing only in RTI and Traffic laws. (in INDIA)
    
    **Instructions:**
    A. If the user's query is NOT in English (the detected language code is {original_lang_code}), first internally translate the query to English for better processing.
    B. Use the following 'Context' to answer the current 'Question'.
    C. If the question is not about Indian Traffic or RTI laws, you MUST politely respond by asking them to focus on those topics.
    D. The FINAL ANSWER MUST BE RETURNED ENTIRELY IN THE USER'S ORIGINAL LANGUAGE (code: {original_lang_code}).
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # Construct the Final Message List (History + Current User Prompt)
    final_message_list = chat_history_messages + [HumanMessage(content=system_instruction)]

    try:
        response = llm.invoke(final_message_list)
        return response.content
    except Exception as e:
        return f"An error occurred with the LLM call: {e}"
        
# 4. Streamlit Initialization and Layout
@st.cache_resource
def setup_rag_components():
    """Initializes and caches the RAG components (LLM, Embeddings, Vector Store, Retriever)."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=2000,
    )

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore(persist_directory="data/vector_store") 
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    
    if vectorstore.collection.count() == 0:
        st.warning("⚠️ **Warning**: The vector store is empty. Please run your data loading script first to populate the vector store at the **directory** `./data/vector_store`.")

    return llm, rag_retriever

# --- Main App Execution ---

st.set_page_config(page_title="LawGPT", page_icon="⚖️", layout="wide")

st.title("⚖️ LawBot")
st.subheader("Your AI legal helper for Indian Traffic and RTI laws.")

# Initialize components (Cached)
llm, rag_retriever = setup_rag_components()

# Initialize session state flags
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am LawGPT. I can help you with questions about **Indian Traffic and RTI laws**. How can I assist you today?"}]
if "retrieved_count" not in st.session_state:
    st.session_state.retrieved_count = 0
if "log_reset_needed" not in st.session_state:
    st.session_state.log_reset_needed = False 
if 'admin_logged_in' not in st.session_state: # <-- Initialize new state flag
    st.session_state.admin_logged_in = False 

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about Traffic or RTI law..."):
    # 1. Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get assistant response (MUST be inside the chat_message block)
    with st.chat_message("assistant"):
        with st.spinner("LawGPT is drafting the legal advice..."): 
            
            response_content = rag_simple(
                prompt, 
                rag_retriever, 
                llm, 
                top_k=3,
                history=st.session_state.messages 
            )
            
            st.markdown(response_content) 
            
    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    
    # 4. 📝 LOG INTERACTION AFTER RESPONSE IS GENERATED
    log_user_interaction(
        query=prompt, 
        response=response_content,
        retriever_count=st.session_state.retrieved_count
    )

#FEATURE 1: Admin Log Viewer with Password Protection
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("Admin Log Viewer (Usage) 📊"):
    
    if not st.session_state.admin_logged_in:
        
        # Display login form
        st.markdown("### Admin Login")
        with st.form("admin_login_form"):
            admin_pass = st.text_input("Password:", type="password", key="admin_pass_input")
            submitted = st.form_submit_button("Login")
        
            if submitted:
                if admin_pass == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.success("Login Successful!")
                    st.rerun() # Rerun to display logs
                else:
                    st.error("Wrong password.")
                
    else:
        # --- LOGS DISPLAY LOGIC (Only runs if logged in) ---
        st.markdown("### User Query Log")
        
        # Logout Button
        if st.button("Logout", key="logout_btn"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        check_and_reset_log(LOG_FILE_PATH) 

        if os.path.exists(LOG_FILE_PATH):
            try:
                logs_df = pd.read_csv(LOG_FILE_PATH)
                st.session_state.log_reset_needed = False 
                
                st.dataframe(logs_df.tail(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔴 Clear All Logs", use_container_width=True):
                        st.session_state.log_reset_needed = True
                        st.rerun()
                
                with col2:
                    st.download_button(
                        label="Download Full Log CSV",
                        data=logs_df.to_csv(index=False).encode('utf-8'), 
                        file_name="lawgpt_user_logs.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
            except pd.errors.EmptyDataError:
                st.info("Log file exists but is currently empty.")
            except Exception as e:
                st.error(
                    "❌ **Log Corruption Error:** The log will be automatically reset on your next interaction."
                )
                st.session_state.log_reset_needed = True
                st.exception(e)
        else:

            st.info("No log file found yet.")
