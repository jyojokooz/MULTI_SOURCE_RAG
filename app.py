import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Keys
load_dotenv()

# --- Smart History Management ---
HISTORY_FILE = "chat_history.json"

def load_history():
    """Loads all previously asked questions and answers from the file securely."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Safely only load items that have the correct format to prevent crashes
                valid_history = [item for item in data if "question" in item and "answer" in item]
                return valid_history
        except Exception:
            return []
    return []

def save_history(history_list):
    """Saves the Q&A list securely to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_list, f, indent=4)
# -------------------------------------

# 2. Setup Streamlit Page (The UI)
st.set_page_config(page_title="NanoPhysics AI Assistant", page_icon="🔬", layout="wide")
st.title("🔬 NanoPhysics & Nanoelectronics AI Assistant")
st.markdown("Ask me any question, and I will search across your nanophysics textbooks to write a **massive, highly detailed academic essay!**")

# 3. Cache the heavy AI models
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = PineconeVectorStore(index_name="textbook-index", embedding=embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    # INCREASED TEMPERATURE AND TOKENS FOR MASSIVE ESSAYS
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.5,  # Increased from 0.3 so the AI elaborates much more
        max_tokens=4000   # Increased to ensure giant essays don't get cut off
    )

vectorstore = load_vectorstore()
llm = load_llm()

# 4. Setup the AI Brain (SUPERCHARGED PROMPT FOR MAXIMUM ESSAY SIZE)
prompt = ChatPromptTemplate.from_template("""
You are an expert, highly verbose, and professional Physics Professor. 
Your goal is to write a MASSIVE, comprehensive, long-form academic essay to answer the user's question, using the retrieved textbook context as your primary foundation.

Guidelines for your response:
1. **Massive Essay Length:** Your response MUST be extremely long, detailed, and expansive (minimum of 800 to 1000+ words). Write multiple thick, exhaustive paragraphs. Under NO circumstances should you provide a short, brief, or basic summary. Act as if you are writing a full chapter for a textbook.
2. **Elaborate Deeply:** Use the retrieved context as your factual anchor. However, you MUST use your internal expert physics knowledge to deeply explain, expand, and contextualize EVERY formula, principle, and derivation mentioned. Assume the reader needs step-by-step hand-holding through the complex physics.
3. **Logical Structure:** Use bold headings, bullet points, and numbered lists where appropriate, but ensure they are padded with massive, detailed paragraphs of explanatory text connecting the ideas.
4. **No Dead Ends:** If the context only has partial information, thoroughly explain what you do have, and then naturally fill in the physics gaps using your expert knowledge to ensure a complete, dissertation-length conceptual understanding.

Context:
{context}

Question: 
{input}

Massive Detailed Essay Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
# Search the database for the top 8 most relevant paragraphs
retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 
rag_chain = create_retrieval_chain(retriever, document_chain)

# 5. Initialize State Variables
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "current_messages" not in st.session_state:
    st.session_state.current_messages = []

# --- ChatGPT-Style Sidebar ---
with st.sidebar:
    st.header("📚 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_messages = []
        st.rerun()
        
    st.divider()
    st.markdown("**Previous Questions:**")
    
    for idx, item in enumerate(reversed(st.session_state.history)):
        q_text = item["question"]
        btn_text = (q_text[:35] + "...") if len(q_text) > 35 else q_text
        
        if st.button(f"📝 {btn_text}", key=f"hist_{idx}", help=q_text):
            st.session_state.current_messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
            st.rerun()
# ----------------------------------

# 6. Display ONLY the currently active conversation
for message in st.session_state.current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. React to the User's Question
if user_input := st.chat_input("Ask a physics question..."):
    
    # SMART SEARCH: Check if this question was already asked!
    existing_item = next((item for item in st.session_state.history if item["question"].strip().lower() == user_input.strip().lower()), None)
    
    if existing_item:
        st.session_state.current_messages = [
            {"role": "user", "content": existing_item["question"]},
            {"role": "assistant", "content": existing_item["answer"]}
        ]
        st.rerun()
        
    else:
        st.session_state.current_messages = [{"role": "user", "content": user_input}]
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Writing a massive academic essay based on your textbooks... (this may take a few seconds)"):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                
                # Source Citations
                sources = []
                for doc in response["context"]:
                    if 'page' in doc.metadata and 'source' in doc.metadata:
                        book_name = os.path.basename(doc.metadata['source'])
                        page_num = doc.metadata['page'] + 1
                        sources.append(f"**{book_name}** (Page {page_num})")
                
                if sources:
                    unique_sources = list(set(sources))
                    answer += f"\n\n---\n**Sources Used:** {', '.join(unique_sources)}"
                
                st.markdown(answer)
        
        # 1. Update the screen
        st.session_state.current_messages.append({"role": "assistant", "content": answer})
        
        # 2. Auto-Save to the global history list
        st.session_state.history.append({
            "question": user_input,
            "answer": answer
        })
        
        # 3. Permanently save to the JSON file
        save_history(st.session_state.history)