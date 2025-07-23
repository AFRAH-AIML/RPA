import os
import hashlib
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import numpy as np

# Global embedding cache
embedding_cache = {}

# Configuration
CHUNK_SIZE = 512  # Optimal chunk size for better context
CHUNK_OVERLAP = 50  # Overlap between chunks for better continuity

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a lightweight embedding model"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def load_document(self, file_path_or_file) -> Dict[str, Any]:
        """Load document and return text with metadata"""
        if hasattr(file_path_or_file, 'read'):  # Streamlit uploaded file
            file_name = file_path_or_file.name
            file_content = file_path_or_file.read()
            
            if file_name.endswith('.pdf'):
                return self._extract_pdf_text(file_content, file_name)
            elif file_name.endswith('.docx'):
                return self._extract_docx_text(file_content, file_name)
            elif file_name.endswith('.txt'):
                return {
                    'text': file_content.decode('utf-8'),
                    'filename': file_name,
                    'pages': [{'page_num': 1, 'text': file_content.decode('utf-8')}]
                }
        else:  # File path
            if file_path_or_file.endswith('.pdf'):
                with open(file_path_or_file, 'rb') as f:
                    return self._extract_pdf_text(f.read(), os.path.basename(file_path_or_file))
            elif file_path_or_file.endswith('.docx'):
                with open(file_path_or_file, 'rb') as f:
                    return self._extract_docx_text(f.read(), os.path.basename(file_path_or_file))
            elif file_path_or_file.endswith('.txt'):
                with open(file_path_or_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    return {
                        'text': text,
                        'filename': os.path.basename(file_path_or_file),
                        'pages': [{'page_num': 1, 'text': text}]
                    }
    
    def _extract_pdf_text(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from PDF with page information"""
        from io import BytesIO
        reader = PdfReader(BytesIO(file_content))
        
        pages = []
        full_text = ""
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            pages.append({
                'page_num': page_num,
                'text': page_text
            })
            full_text += page_text + "\n"
        
        return {
            'text': full_text,
            'filename': filename,
            'pages': pages
        }
    
    def _extract_docx_text(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        from io import BytesIO
        doc = docx.Document(BytesIO(file_content))
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return {
            'text': text,
            'filename': filename,
            'pages': [{'page_num': 1, 'text': text}]  # DOCX treated as single page
        }
    
    def create_chunks_with_metadata(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks with proper metadata for source tracking"""
        chunks = []
        
        for page_info in document['pages']:
            page_text = page_info['text']
            page_num = page_info['page_num']
            
            # Split text into sentences for better chunking
            sentences = self._split_into_sentences(page_text)
            
            current_chunk = ""
            chunk_start_sentence = 0
            
            for i, sentence in enumerate(sentences):
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk + sentence) > CHUNK_SIZE and current_chunk:
                    # Create chunk with metadata
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'filename': document['filename'],
                        'page_num': page_num,
                        'chunk_id': f"{document['filename']}_page{page_num}_chunk{len(chunks)+1}",
                        'sentence_range': f"{chunk_start_sentence+1}-{i}"
                    }
                    chunks.append(chunk_data)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(sentences, max(0, i-3), i)
                    current_chunk = overlap_text + sentence
                    chunk_start_sentence = max(0, i-3)
                else:
                    current_chunk += sentence
            
            # Add final chunk if there's remaining text
            if current_chunk.strip():
                chunk_data = {
                    'text': current_chunk.strip(),
                    'filename': document['filename'],
                    'page_num': page_num,
                    'chunk_id': f"{document['filename']}_page{page_num}_chunk{len(chunks)+1}",
                    'sentence_range': f"{chunk_start_sentence+1}-{len(sentences)}"
                }
                chunks.append(chunk_data)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]
    
    def _get_overlap_text(self, sentences: List[str], start: int, end: int) -> str:
        """Get overlap text for chunk continuity"""
        return " ".join(sentences[start:end]) + " "
    
    def embed_text(self, text: str) -> List[float]:
        """Create embedding for text with caching"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        if text_hash in embedding_cache:
            return embedding_cache[text_hash]
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            embedding = embedding.tolist()
            embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            print(f"Embedding failed for text: {text[:50]}... Error: {e}")
            return None

def create_and_store_embeddings(client: QdrantClient, processor: DocumentProcessor, 
                               chunks: List[Dict[str, Any]], collection_name: str):
    """Create and store embeddings in Qdrant"""
    
    # Check if collection exists, if not create it
    try:    
        client.get_collection(collection_name)
    except:
        # Get embedding dimension from a sample
        sample_embedding = processor.embed_text("sample text")
        if sample_embedding is None:
            raise ValueError("Failed to create sample embedding")
            
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(sample_embedding), 
                distance=models.Distance.COSINE
            )
        )
    
    # Prepare points for insertion
    points = []
    for chunk in chunks:
        embedding = processor.embed_text(chunk['text'])
        if embedding is not None:
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'text': chunk['text'],
                    'filename': chunk['filename'],
                    'page_num': chunk['page_num'],
                    'chunk_id': chunk['chunk_id'],
                    'sentence_range': chunk['sentence_range']
                }
            )
            points.append(point)
    
    if points:
        # Insert in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
        print(f"Stored {len(points)} chunks in collection '{collection_name}'")

def search_collection(client: QdrantClient, processor: DocumentProcessor, 
                     query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search collection and return results with metadata"""
    
    query_embedding = processor.embed_text(query)
    if query_embedding is None:
        return []
    
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for hit in search_result:
            results.append({
                'text': hit.payload['text'],
                'filename': hit.payload['filename'],
                'page_num': hit.payload['page_num'],
                'chunk_id': hit.payload['chunk_id'],
                'sentence_range': hit.payload['sentence_range'],
                'score': hit.score
            })
        
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []