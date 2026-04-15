"""
Vector Store Module
Handles document embedding and FAISS vector database operations
"""
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path


class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vector_db_path: str = "./data/vector_db"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, documents: List[Dict[str, str]]):
        """Build FAISS index from documents"""
        # Extract text from documents
        texts = [doc['text'] for doc in documents]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Build FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        # Store documents
        self.documents = documents
        
        print(f" Built index with {len(documents)} documents")
    
    def add_documents(self, new_documents: List[Dict[str, str]]):
        """Add new documents to existing index"""
        if self.index is None:
            self.build_index(new_documents)
            return
        
        texts = [doc['text'] for doc in new_documents]
        embeddings = self.create_embeddings(texts)
        
        self.index.add(embeddings)
        self.documents.extend(new_documents)
        
        print(f" Added {len(new_documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """Search for most similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['score'] = float(dist)
                results.append(result)
        
        return results
    
    def save(self, name: str = "vector_store"):
        """Save index and documents to disk"""
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        index_path = self.vector_db_path / f"{name}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents
        docs_path = self.vector_db_path / f"{name}_docs.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f" Saved vector store to {self.vector_db_path}")
    
    def load(self, name: str = "vector_store"):
        """Load index and documents from disk"""
        index_path = self.vector_db_path / f"{name}.index"
        docs_path = self.vector_db_path / f"{name}_docs.pkl"
        
        if not index_path.exists() or not docs_path.exists():
            print("No saved vector store found")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load documents
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f" Loaded vector store with {len(self.documents)} documents")
        return True
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.documents = []
        print(" Vector store cleared")
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the vector store"""
        if self.index is None:
            return {
                'total_documents': 0,
                'dimension': self.dimension,
                'index_type': None
            }
        
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'index_type': 'FAISS IndexFlatL2'
        }


if __name__ == "__main__":
    # Test the vector store
    vector_store = VectorStore()
    
    # Test documents
    test_docs = [
        {'text': 'This is a test document about machine learning.', 'chunk_id': 0},
        {'text': 'Another document about natural language processing.', 'chunk_id': 1},
    ]
    
    vector_store.build_index(test_docs)
    results = vector_store.search("machine learning", top_k=1)
    print(f"Search results: {results}")
