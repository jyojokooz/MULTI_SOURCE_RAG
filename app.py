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
st.markdown("Ask me any question, and I will search across your nanophysics textbooks to find a detailed, step-by-step answer!")

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
    # Connect to Llama-3.1 via Groq (Temperature set to 0.2 for better explanations)
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

vectorstore = load_vectorstore()
llm = load_llm()

# 4. Setup the AI Brain (Optimized for detailed Physics explanations)
prompt = ChatPromptTemplate.from_template("""
You are an expert, highly detailed, and professional Physics Professor and academic tutor. 
Your goal is to provide comprehensive, thorough, and step-by-step explanations to the user's question using the retrieved context from the textbooks.

Guidelines for your response:
1. **Be Highly Detailed:** Do not give short or brief answers. Explain the underlying physics principles, concepts, formulas, and derivations if they are present in the context.
2. **Logical Structure:** Structure your answer cleanly. Use bold headings, bullet points, and numbered lists to break down the explanation "question-wise" or step-by-step.
3. **Accuracy:** Ground your explanation strictly in the retrieved context. If some specific details are completely missing, explain what you *can* find in the context thoroughly, rather than giving up.

Context:
{context}

Question: 
{input}

Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Search the database for the top 8 most relevant paragraphs (Increased for deeper context)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 
rag_chain = create_retrieval_chain(retriever, document_chain)

# 5. Build the Chat History UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. React to the User's Question
if user_input := st.chat_input("Ask a physics question..."):
    # Show what the user typed
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show a loading spinner while AI thinks
    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing nanophysics textbooks..."):
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