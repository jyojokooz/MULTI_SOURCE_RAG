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
st.markdown("Ask me any question, and I will search across your nanophysics textbooks to write a highly detailed, essay-style answer!")

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
    # Upgraded to a 70B parameter model for deep reasoning and detailed essay generation
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.3, # Slightly increased for better creative elaboration
        max_tokens=3000  # Ensures the AI has enough room to write long essays
    )

vectorstore = load_vectorstore()
llm = load_llm()

# 4. Setup the AI Brain (Optimized for long-form Academic Essays)
prompt = ChatPromptTemplate.from_template("""
You are an expert, highly detailed, and professional Physics Professor and academic tutor. 
Your goal is to write a comprehensive, long-form academic essay to answer the user's question, using the retrieved textbook context as your primary foundation.

Guidelines for your response:
1. **Essay Format & Length:** Write in a detailed, exhaustive, and expansive format. Your response should read like a thorough textbook chapter or academic essay. Do NOT give brief or basic summaries. 
2. **Elaborate Context:** Use the retrieved context as your factual anchor. However, you are highly encouraged to use your internal expert physics knowledge to deeply explain, elaborate on, and contextualize the formulas, principles, and derivations found in the context.
3. **Logical Structure:** Use bold headings, bullet points, and numbered lists where appropriate, but ensure there are thick, detailed paragraphs of explanatory text connecting the ideas.
4. **No Dead Ends:** If the context only has partial information, thoroughly explain what you do have, and naturally fill in the physics gaps to ensure the user gets a complete conceptual understanding.

Context:
{context}

Question: 
{input}

Detailed Essay Answer:
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
        with st.spinner("Writing a detailed academic essay based on your textbooks..."):
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
                answer += f"\n\n---\n**Sources Used:** {', '.join(unique_sources)}"
            
            st.markdown(answer)
    
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})