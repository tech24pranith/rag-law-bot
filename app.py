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
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage 

# --- External RAG Dependencies ---
from sentence_transformers import SentenceTransformer
import chromadb

# Suppress warnings from Sentence Transformers/ChromaDB initialization
warnings.filterwarnings("ignore")

# Define the log file path
LOG_FILE_PATH = "user_logs.csv"

# Load environment variables (assumes GEMINI_API_KEY is in a .env file)
load_dotenv()

# ==============================================================================
# LOGGING AND CLEANUP FUNCTIONS
# ==============================================================================

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
    
    # Create the log entry 
    log_entry = [
        timestamp,
        query,
        response,
        retriever_count
    ]
    
    file_exists = os.path.exists(LOG_FILE_PATH)
    
    try:
        # Use 'a' for append mode, 'newline=""' is critical for cross-platform CSV writing
        with open(LOG_FILE_PATH, 'a', encoding='utf-8', newline="") as f:
            writer = csv.writer(f) 
            
            if not file_exists or os.stat(LOG_FILE_PATH).st_size == 0:
                # Write header row if file is new or empty
                writer.writerow(["Timestamp", "Query", "Response", "Retrieved_Chunks"])
            
            # Write the data row
            writer.writerow(log_entry)
            
    except Exception:
        # Fail silently here to prevent disrupting the user chat flow
        pass 

# ==============================================================================
# 2. RAG Component Classes (ChromaDB Path Fixed)
# ==============================================================================

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

# ==============================================================================
# 3. RAG Execution Function (with Memory)
# ==============================================================================

def rag_simple(query: str, retriever: RAGRetriever, llm: ChatGoogleGenerativeAI, top_k: int = 3, history: List[Dict[str, str]] = None) -> str:
    """
    Performs a single-turn RAG query, using history for better context awareness.
    """
    
    # Retrieve the context (This operation is fast as the embedding model is cached)
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    # Store retrieved count for logging
    st.session_state.retrieved_count = len(results)

    # Format the Conversation History for the LLM
    chat_history_messages = []
    if history:
        for message in history:
            if message['role'] == 'user':
                chat_history_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                chat_history_messages.append(AIMessage(content=message['content']))
    
    # Define the System Prompt
    system_instruction = f"""You are a legal helper in India with high knowledge in your field, specializing only in RTI and Traffic laws. (in INDIA)
    
    RULES:
    1. If the user asks a question *not* related to Indian Traffic or RTI laws, you MUST politely respond by asking them to focus on those topics.
    2. Use the following 'Context' to answer the current 'Question' concisely.
    3. Try to generate a short, clear answer (target 90 words).
    4. If 'Context' is not found, rely on your general knowledge of Indian RTI and Traffic laws and answer as a professional lawyer.
    5. If the question is based on general examples, briefly explain a similar real-life case if relevant to the law.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # Construct the Final Message List (History + Current User Prompt)
    final_message_list = chat_history_messages + [HumanMessage(content=system_instruction)]

    try:
        # LLM invocation is done within the spinner block in the main app layout
        response = llm.invoke(final_message_list)
        return response.content
    except Exception as e:
        return f"An error occurred with the LLM call: {e}"

# ==============================================================================
# 4. Streamlit Initialization and Layout
# ==============================================================================

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
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am LawGPT. I can help you with questions about **Indian Traffic and RTI laws**. How can I assist you today?"}
    ]
if "retrieved_count" not in st.session_state:
    st.session_state.retrieved_count = 0
if "log_reset_needed" not in st.session_state:
    st.session_state.log_reset_needed = False 


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
        # The spinner ensures the "thinking" message stays visible until the LLM returns
        with st.spinner("LawGPT is drafting the legal advice..."): 
            
            response_content = rag_simple(
                prompt, 
                rag_retriever, 
                llm, 
                top_k=3,
                history=st.session_state.messages 
            )
            
            # Display the final response
            st.markdown(response_content) 
            
    # 3. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    
    # 4. 📝 LOG INTERACTION AFTER RESPONSE IS GENERATED
    log_user_interaction(
        query=prompt, 
        response=response_content,
        retriever_count=st.session_state.retrieved_count
    )

# ------------------------------------------------------------------------------
## 💻 Admin Log Viewer
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("Admin Log Viewer (Usage) 📊"):
    
    # Check and delete log file if corruption was detected in the last run
    check_and_reset_log(LOG_FILE_PATH) 

    if os.path.exists(LOG_FILE_PATH):
        try:
            # Read the logs and display the table
            logs_df = pd.read_csv(LOG_FILE_PATH)
            
            # Reset the cleanup flag on successful read
            st.session_state.log_reset_needed = False 
            
            st.markdown("### User Query Log")
            st.dataframe(logs_df.tail(10), use_container_width=True) # Show last 10 entries
            if st.button("🔴 Clear All Logs"):
            # Set the flag to true and rerun the app
                st.session_state.log_reset_needed = True
                st.rerun()
            st.download_button(
                label="Download Full Log CSV",
                data=logs_df.to_csv(index=False).encode('utf-8'), 
                file_name="lawgpt_user_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except pd.errors.EmptyDataError:
            st.info("Log file exists but is currently empty.")
            st.session_state.log_reset_needed = False
        
        except Exception as e:
            # 🔴 If we fail to read the log due to corruption, set a flag 
            # to reset it on the next user interaction (app rerun).
            st.error(
                "❌ **Log Corruption Error:** The log file format is damaged. "
                "The file will be automatically reset on your next app interaction."
            )
            st.session_state.log_reset_needed = True
            st.exception(e)
    else:
        st.info("No log file found yet. Ask a question to generate the first entry.")