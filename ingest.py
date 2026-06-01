import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load API keys from the .env file
load_dotenv()

# Verify Pinecone API key is loaded
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not found. Please check your .env file.")

def process_and_upload_pdf():
    print("--- Starting the Ingestion Process ---")
    
    # 2. Load the large PDF
    pdf_path = "textbook.pdf" # Make sure this matches your file name!
    print(f"Step 1: Loading '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Success! Loaded {len(documents)} pages from the PDF.")

    # 3. Split the text into manageable chunks
    # We split into 1000-character chunks with a 200-character overlap 
    # to ensure sentences aren't cut in half and context is preserved.
    print("\nStep 2: Splitting text into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Success! Split the document into {len(chunks)} smaller chunks.")

    # 4. Initialize HuggingFace Embeddings (100% Free)
    # This model outputs vectors with exactly 384 dimensions (which matches our Pinecone index)
    print("\nStep 3: Loading Embedding Model...")
    print("(If this is your first time, it may take a minute to download the free model to your PC)")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 5. Upload permanently to Pinecone
    index_name = "textbook-index" # Make sure this matches the index name you created in Pinecone!
    print(f"\nStep 4: Uploading vectors to Pinecone cloud index '{index_name}'...")
    print("WARNING: For a 1000-page PDF, this may take 5 to 15 minutes. Please let it run...")
    
    PineconeVectorStore.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    
    print("\nSUCCESS! Your massive textbook has been permanently saved to the Pinecone cloud!")
    print("You NEVER have to run this script again for this textbook.")

if __name__ == "__main__":
    process_and_upload_pdf()