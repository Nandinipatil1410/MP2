"""
Document Processing Module
Handles PDF, DOCX, and TXT file processing and chunking
"""
from typing import List, Dict
import PyPDF2
from docx import Document
from pathlib import Path
import re


class DocumentProcessor:
    """Process various document types and extract text"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error processing PDF: {e}")
        return text
    
    def process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error processing DOCX: {e}")
        return text
    
    def process_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error processing TXT: {e}")
            return ""
    
    def process_file(self, file_path: str) -> str:
        """Process file based on extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(file_path)
        elif extension == '.docx':
            return self.process_docx(file_path)
        elif extension == '.txt':
            return self.process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_idx': i
            })
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def process_and_chunk(self, file_path: str) -> List[Dict[str, str]]:
        """Complete pipeline: process file and create chunks"""
        text = self.process_file(file_path)
        if not text:
            return []
        
        chunks = self.chunk_text(text)
        
        # Add metadata
        for chunk in chunks:
            chunk['source'] = Path(file_path).name
        
        return chunks


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    print("Document Processor module loaded successfully!")
