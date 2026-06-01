import os
import time
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
    
    folder_path = "books"
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} textbooks to process!\n")

    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    index_name = "textbook-index"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # NEW: Keep track of what we already uploaded so we can resume if it crashes!
    tracker_file = "uploaded_books.txt"
    if os.path.exists(tracker_file):
        with open(tracker_file, "r", encoding="utf-8") as f:
            uploaded_books = set(f.read().splitlines())
    else:
        uploaded_books = set()

    for file_name in pdf_files:
        # If we already uploaded this book in a previous run, skip it!
        if file_name in uploaded_books:
            print(f"⏭️ Skipping '{file_name}' (Already uploaded!)")
            continue

        pdf_path = os.path.join(folder_path, file_name)
        print(f"\n📚 Loading '{file_name}'...")
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            chunks = text_splitter.split_documents(documents)
            print(f"   -> Split into {len(chunks)} chunks. Uploading to Pinecone...")
            
            PineconeVectorStore.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                index_name=index_name
            )
            print(f"   -> Successfully saved '{file_name}' to cloud!")
            
            # Mark as successfully uploaded so we never do it again
            with open(tracker_file, "a", encoding="utf-8") as f:
                f.write(file_name + "\n")
            uploaded_books.add(file_name)
            
            # Pause for 2 seconds to avoid overwhelming Pinecone's free tier
            time.sleep(2)
            
        except Exception as e:
            print(f"\n❌ Error uploading '{file_name}': {e}")
            print("Pinecone disconnected. Stopping script. Just run the script again to resume!")
            break 

    print("\nSUCCESS! Ingestion process finished.")

if __name__ == "__main__":
    process_multiple_pdfs()