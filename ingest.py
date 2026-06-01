import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not found. Please check your .env file.")

def process_multiple_pdfs():
    print("--- Starting the Multi-Book Ingestion Process ---")
    
    # Target the folder where your 20 books are
    folder_path = "books"
    
    # Get a list of all PDF files in that folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} textbooks to process!\n")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    index_name = "textbook-index"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Loop through each book one by one so we don't crash your computer's RAM
    for file_name in pdf_files:
        pdf_path = os.path.join(folder_path, file_name)
        print(f"📚 Loading '{file_name}'...")
        
        # 1. Load this specific book
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 2. Split this book
        chunks = text_splitter.split_documents(documents)
        print(f"   -> Split into {len(chunks)} chunks. Uploading to Pinecone...")
        
        # 3. Upload this book to the cloud
        PineconeVectorStore.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            index_name=index_name
        )
        print(f"   -> Successfully saved '{file_name}' to cloud!\n")
        
    print("SUCCESS! All textbooks have been uploaded to Pinecone.")

if __name__ == "__main__":
    process_multiple_pdfs()