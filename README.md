# 🔬 NanoPhysics & Nanoelectronics AI Assistant
An AI-powered web application that allows users to perform semantic search and "chat" with a massive library of 21 advanced textbooks spanning Nanophysics, Solid-State Physics, Nanoelectronics, and Nanotechnology.

## 🚀 Live Demo
[Click here to view the live app!](https://multisourcerag-dypf8vhybzrixpbgpxxtbq.streamlit.app/)

## 🛠️ Architecture & Tech Stack
This project utilizes Retrieval-Augmented Generation (RAG) to ground the Large Language Model's answers strictly in verified textbook data, minimizing the risk of AI hallucinations.
* **Frontend:** Streamlit
* **LLM:** Meta Llama-3.1-8B (via Groq Cloud for ultra-fast, low-latency inference)
* **Vector Database:** Pinecone
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Orchestration:** LangChain

## 📚 Topics Covered in the Library
The vector database contains parsed, chunked, and embedded contents from 21 distinct textbooks, covering:
* Quantum Transport & Mesoscopic Physics
* Carbon Nanotube Electronic Devices
* Spintronics & Nanomagnetism
* Solid-State Nanostructures & Quantum Dots
* Nanophotonics & Optical Properties of Nanocrystals
* Nanomaterials Synthesis & Nanoparticles

## ✨ Features
* **Multi-Book Semantic Search:** Queries across 21 textbooks (thousands of pages chunked into vector embeddings) simultaneously in milliseconds.
* **Automated Book & Page Citation:** Displays the exact textbook filename and original page numbers used to construct each answer for verification.
* **Smart Resume Ingestion:** The ingestion pipeline (`ingest.py`) tracks successfully vectorized PDFs using a local tracker to gracefully handle network drops or API rate-limits without re-uploading from scratch.
* **Chat Memory:** Contextual chat window history for conversational follow-up questions.

## ⚙️ How It Works (Local Setup)

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

   Install Dependencies

    pip install -r requirements.txt
    3. Configure Environment Variables
    Create a .env file in the root directory:

    GROQ_API_KEY="your_groq_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    4. Vectorize Textbooks
    Place your PDF files in a /books folder and run the smart ingestion script:
 
    python ingest.py
    5. Run the Application
    
    streamlit run app.py
