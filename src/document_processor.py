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
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with multiple fallbacks and OCR"""
        text = ""
        try:
            # 1. Try pdfplumber for robust text and table extraction
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    # Extract standard text
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    # Extract tables separately and format them cleanly
                    tables = page.extract_tables()
                    if tables:
                        text += f"\n--- Tables Found on Page (Source: {Path(file_path).name}) ---\n"
                        for table in tables:
                            if not table or not any(table): continue
                            # Try to identify header row (usually first non-empty row)
                            headers = []
                            for row in table:
                                if row and any(row):
                                    headers = [str(h).strip().replace('\n', ' ') for h in row if h]
                                    break
                            
                            current_section = ""
                            for row in table:
                                # Clean up cell content and join with '|'
                                clean_row = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                                if not any(clean_row): continue
                                
                                # Detect Section headers (e.g., "I Income", "II Expenditure")
                                non_empty = [c for c in clean_row if c]
                                if len(non_empty) == 1:
                                    item = non_empty[0]
                                    if re.match(r'^[IVX]+[\.\s]', item):
                                        current_section = re.sub(r'^[IVX]+[\.\s]+', '', item).strip()
                                        text += f"--- Section: {current_section} ---\n"
                                        continue

                                # Contextualize "Total" rows
                                if "total" in clean_row[0].lower() and current_section:
                                    if clean_row[0].strip().lower() == "total":
                                        clean_row[0] = f"Total {current_section}"
                                
                                # Format as key-value if possible, else standard row
                                row_str = ""
                                if len(headers) == len(clean_row):
                                    row_str = " | ".join([f"{h}: {v}" if h and v else v for h, v in zip(headers, clean_row)])
                                else:
                                    row_str = " | ".join(clean_row)
                                
                                if current_section and current_section.lower() not in row_str.lower():
                                    row_str = f"[{current_section}] {row_str}"

                                text += row_str + "\n"
                            text += "\n"
            
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
    
    def chunk_text(self, text: str, source_name: str = "") -> List[Dict[str, str]]:
        """Split text into overlapping word-based chunks while preserving newlines and injecting context"""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        chunks = []
        words = text.split(' ')
        
        # Inject source context into every chunk to aid retrieval and grounding
        context_prefix = f"[Document: {source_name}] " if source_name else ""
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_body = ' '.join(chunk_words)
            chunk_text = context_prefix + chunk_body
            
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
        
        source_name = Path(file_path).name
        chunks = self.chunk_text(text, source_name=source_name)
        
        # Add metadata
        for chunk in chunks:
            chunk['source'] = source_name
        
        return chunks


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    print("Document Processor module loaded successfully!")
