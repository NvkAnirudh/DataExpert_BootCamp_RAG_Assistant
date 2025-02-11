from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

def load_and_process_pdfs(data_dir):
    """Load and process PDFs from the specified directory."""
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks, db_dir):
    """Create a vector store (ChromaDB) from the document chunks."""
    # clear existing vector store if it exists
    if os.path.exists(db_dir):
        print(f'Deleting existing vector store at {db_dir}')
        shutil.rmtree(db_dir)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

    # Create and persist the Chroma vector store
    print('Creating Chroma vector store...')
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_dir)
    vectorstore.persist()
    return vectorstore

def main():
    # Define directories
    data_dir = os.path.join(os.path.dirname(__file__),"data")
    db_dir = os.path.join(os.path.dirname(__file__),"chroma_db")

    # Process PDFs
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Loaded {len(chunks)} chunks from PDFs.")

    # Create embeddings (vector store)
    print("Creating embeddings (vector store)...")
    vectorstore = create_vectorstore(chunks, db_dir)
    print(f"Vector store created with {len(vectorstore)} embeddings and persisted at {db_dir}.")

if __name__ == "__main__":
    main()