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
        """Extract text from PDF file with multiple fallbacks and OCR"""
        text = ""
        try:
            # 1. Try PyPDF2 (fastest)
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + "\n"
            
            # 2. Check if extraction is "suspiciously" low quality
            # (e.g., only copyright notice from a multi-page doc)
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            unique_lines = set(lines)
            
            # Heuristic: If it's a multi-page doc but very few unique lines, it's likely scanned or watermarked
            is_suspicious = False
            if num_pages > 1 and len(unique_lines) < num_pages: 
                is_suspicious = True
            if "Copyright" in text and len(unique_lines) < 3:
                is_suspicious = True
                
            if is_suspicious or not text.strip():
                print(f"  ⚠️  Standard extraction failed or returned low-quality text for {Path(file_path).name}. Trying pdfminer...")
                from pdfminer.high_level import extract_text as miner_extract
                text = miner_extract(file_path)
            
            # 3. Final Fallback: OCR for scanned documents
            # If still suspiciously short text after pdfminer
            if len(text.strip()) < 100 * num_pages: # Heuristic: less than 100 chars per page
                print(f"  🔍 Document {Path(file_path).name} appears to be scanned. Running OCR (this may take a while)...")
                try:
                    from pdf2image import convert_from_path
                    import easyocr
                    import numpy as np
                    
                    # Initialize reader (lazy load)
                    reader = easyocr.Reader(['en'])
                    
                    # Convert PDF to images
                    images = convert_from_path(file_path)
                    
                    ocr_text = ""
                    for i, image in enumerate(images):
                        # Convert PIL image to numpy array for easyocr
                        img_np = np.array(image)
                        results = reader.readtext(img_np, detail=0)
                        ocr_text += " ".join(results) + "\n"
                    
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        print(f"  ✓ OCR successful for {Path(file_path).name}")
                except ImportError as e:
                    print(f"  ✗ OCR failed: Optional dependencies missing. {e}")
                except Exception as e:
                    print(f"  ✗ OCR failed: {e}")
                    
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
