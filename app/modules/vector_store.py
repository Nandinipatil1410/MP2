import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, vector_db_path="data/vectors/faiss_index"):
        self.vector_db_path = vector_db_path
        # Using a small, efficient local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def process_pdf(self, pdf_path):
        """Loads a PDF, splits it into chunks, and adds to FAISS index."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        if os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        self.vector_store.save_local(self.vector_db_path)
        return len(chunks)

    def get_retriever(self, k=3):
        """Returns a retriever for the vector store."""
        if not self.vector_store and os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None

    def similarity_search(self, query, k=3):
        """Performs a similarity search."""
        if not self.vector_store and os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []
