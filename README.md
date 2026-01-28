#  LawGPT
## 🚀Multilingual RAG-based Legal Assistant ( Traffic & RTI Laws )  

## 🎬Working 
------Video ----------
## 🛠️Tech Stack Used
| Component          | Technology                                |
| ------------------ | ----------------------------------------- |
| UI                 | Streamlit                                 |
| LLM                | Google Gemini (LangChain)                 |
| Embeddings         | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB          | ChromaDB                                  |
| Logging            | CSV + Pandas                              |

## ✨Features
#### 1. Retrieval-Augmented Generation (RAG)

  *  Uses SentenceTransformers for embeddings
  *  Stores and retrieves document embeddings using ChromaDB
  *   Provides context-aware, grounded responses using retrieved legal documents

#### 2. Gemini-powered LLM
  *  Uses Google Gemini (gemini-2.5-flash) via LangChain

#### 3. Conversational Memory
  *  Maintains chat history for better contextual understanding

#### 4. Admin Log Viewer
  *  Password-protected admin panel
  *  Logs ( Timestamp, User query, Model response, Number of retrieved chunks )
  *  Download logs as CSV

#### 5. Domain Restriction
  *  Can answer queries related to Indian Traffic & RTI Laws
  *  Politely refuses unrelated queries

#### 6. Multilingual Support
  * All responses are returned in the user’s original language

## ✅How RAG Works in This Project
1. User asks a question 
2. Query is embedded using SentenceTransformers
3. Relevant legal chunks are retrieved from ChromaDB
4. Gemini generates an answer using: Retrieved context &  Conversation history
5. Final response is returned to the user

## 📊Admin Panel
*  Accessible from the Streamlit sidebar
*  Default password
*  View recent user interactions
*  Download full logs as CSV
*  Clear corrupted or old logs securely
