import pdfplumber
import os
from sentence_transformers import SentenceTransformer
# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
# from chromadb import PersistentClient
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    def __call__(self, input_texts):
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        embeddings = self.model.encode(input_texts, convert_to_numpy=True)
        return embeddings.tolist()


class Embedding_Generation:
    def __init__(self):
        self.custom_embeddings = MyEmbeddingFunction()
        # self.chroma_client = chromadb.HttpClient(
        #     host= os.getenv('CHROMA_DB_HOST'),
        #     port=8000,
        #     # settings=chromadb.config.Settings(allow_reset=True)
        # )
        self.chroma_client = PersistentClient(path="chroma_persistent_storage")

        # Get or create the collection with your custom embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_collection",
            embedding_function=self.custom_embeddings
        )

    def read_documents(self):
        current_wd = os.getcwd()
        data_path = os.path.join(current_wd, "src/data")
        raw_documents = os.listdir(data_path)

        documents = []

        for file_name in raw_documents:
            with pdfplumber.open(os.path.join(data_path, file_name)) as pdf:
                document_text = []
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    
                    # Extract tables separately
                    tables = page.extract_tables()
                    tables_text = []
                    for table in tables:
                        # Handle None values in table cells
                        processed_rows = []
                        for row in table:
                            if any(row):  # Check if row has any content
                                # Replace None with empty string and convert all cells to strings
                                processed_row = [str(cell) if cell is not None else "" for cell in row]
                                processed_rows.append(" | ".join(processed_row))
                        if processed_rows:
                            table_text = "\n".join(processed_rows)
                            tables_text.append(table_text)
                    
                    # Extract headers/footers (if they exist)
                    header = page.within_bbox((0, 0, page.width, 100)).extract_text()
                    
                    # Combine structured content
                    structured_text = f"Page {page_num}:\n"
                    if header:
                        structured_text += f"Header: {header}\n"
                    structured_text += f"Main Content: {text}\n"
                    if tables_text:
                        structured_text += f'Tables: {"".join(tables_text)}'
                    
                    document_text.append(structured_text)
                
                documents.append({
                    "id": file_name,
                    "text": "\n".join(document_text),
                    "metadata": {
                        "source": file_name,
                        "page_count": len(pdf.pages)
                    }
                })
        return documents

    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        # First, split into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk_size
            if current_size + sentence_size > chunk_size and current_chunk:
                # Store the current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Include previous sentences up to chunk_overlap
                for prev_sent in reversed(current_chunk):
                    if overlap_size + len(prev_sent) > chunk_overlap:
                        break
                    overlap_chunk.insert(0, prev_sent)
                    overlap_size += len(prev_sent)
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def chunk_generation(self, documents):
        chunked_documents = []
        for doc in documents:
            chunks = self.split_text(doc["text"])
            print("==== Splitting docs into chunks ====")
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "id": f"{doc['id']}_chunk{i+1}",
                    "text": chunk,
                    "metadata": {
                        "source": doc["metadata"]["source"],
                        "page_count": doc["metadata"]["page_count"],
                        "chunk_number": i+1
                    }
                })

        return chunked_documents

    def generate_embeddings(self, chunked_documents):
        for chunk in chunked_documents:
            self.collection.add(
                ids=[chunk["id"]],
                embeddings=self.custom_embeddings([chunk["text"]]),
                metadatas=[{
                    "source": chunk["metadata"]["source"],
                    "page_count": chunk["metadata"]["page_count"],
                    "chunk_number": chunk["metadata"]["chunk_number"],
                    "text": chunk["text"]
                }],
                documents=[chunk["text"]],
            )

        print(f"Total chunks stored in ChromaDB: {self.collection.count()}")
        return True
