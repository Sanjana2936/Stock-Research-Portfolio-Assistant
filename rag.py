import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "./chroma_db"
KB_FILE = "./knowledge_base/glossary.txt"

def setup_rag_pipeline():
    """
    Initializes the vector database by loading the knowledge base documents.
    Run this once, or whenever the knowledge base is updated.
    """
    print("Loading documents...")
    loader = TextLoader(KB_FILE)
    docs = loader.load()
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    print("Initializing embedding model (this may download a model on first run)...")
    # Using a free local model from sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Creating Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("RAG Pipeline setup complete!")
    return vectorstore

def get_retriever():
    """
    Returns a retriever object for querying the vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

def retrieve_financial_concept(query: str) -> str:
    """
    Retrieves the definition of a financial concept from the RAG knowledge base.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the knowledge base."
        
    # Combine the retrieved document contents
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

if __name__ == "__main__":
    import sys
    if not os.path.exists(DB_DIR):
        print("Vector database not found. Setting up RAG pipeline...")
        setup_rag_pipeline()
    else:
        print("Vector database already exists.")
        
    print("\nTesting retrieval for 'P/E Ratio'...")
    print(retrieve_financial_concept("What does a P/E ratio mean?"))
