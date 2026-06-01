import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Keys
load_dotenv()

# 2. Setup Streamlit Page (The UI)
st.set_page_config(page_title="NanoPhysics AI Assistant", page_icon="🔬")
st.title("🔬 NanoPhysics & Nanoelectronics AI Assistant")
st.markdown("Ask me any question, and I will search across your nanophysics textbooks to find the answer!")

# 3. Cache the heavy AI models so the app doesn't slow down
@st.cache_resource
def load_vectorstore():
    # Connect to the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # Connect to your Pinecone Cloud Database
    vectorstore = PineconeVectorStore(index_name="textbook-index", embedding=embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    # Connect to the free Llama-3.1 model via Groq
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

vectorstore = load_vectorstore()
llm = load_llm()

# 4. Setup the AI Brain (Prompt + Retrieval Chain)
# We strictly tell the AI to ONLY use the provided context.
prompt = ChatPromptTemplate.from_template("""
You are a highly intelligent and professional AI assistant. Use the following retrieved context from the textbooks to answer the user's question. 
If the answer is not contained in the context, just say "I don't know based on the textbooks provided." Do not make up information.

Context:
{context}

Question: 
{input}

Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Search the database for the top 5 most relevant paragraphs
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
rag_chain = create_retrieval_chain(retriever, document_chain)

# 5. Build the Chat History UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. React to the User's Question
if user_input := st.chat_input("Ask a question about your textbooks..."):
    # Show what the user typed
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show a loading spinner while AI thinks
    with st.chat_message("assistant"):
        with st.spinner("Searching across all textbooks..."):
            # Pass the question to our RAG chain
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            # PRO FEATURE: Source Citations (UPDATED FOR MULTIPLE BOOKS)
            sources = []
            for doc in response["context"]:
                if 'page' in doc.metadata and 'source' in doc.metadata:
                    # Extract just the file name (e.g., 'Biology_101.pdf')
                    book_name = os.path.basename(doc.metadata['source'])
                    # Extract page number
                    page_num = doc.metadata['page'] + 1
                    # Combine them
                    sources.append(f"**{book_name}** (Page {page_num})")
            
            # If we found sources, add them to the bottom of the answer
            if sources:
                unique_sources = list(set(sources))
                answer += f"\n\n**Sources Used:** {', '.join(unique_sources)}"
            
            st.markdown(answer)
    
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})