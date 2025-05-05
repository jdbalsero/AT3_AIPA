import os
from dotenv import load_dotenv
from groq import AsyncGroq
from backend.embedding_generation import Embedding_Generation
import streamlit as st
import asyncio

class rag_process:
    def __init__(self):
        load_dotenv()
        self.embedding_class = Embedding_Generation()

    def run_embedding_process(self):

        documents = self.embedding_class.read_documents()
        chunks = self.embedding_class.chunk_generation(documents)
        generation = self.embedding_class.generate_embeddings(chunked_documents=chunks)

        if generation:
            return "Process Complete"
        else:
            return "Error in embedding generation"

    def query_documents(self, question, n_results=2):
        query_embedding = self.embedding_class.custom_embeddings([question])

        results = self.embedding_class.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas"]
        )

        relevant_chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        return relevant_chunks, metadatas

    def generate_response(self, question, relevant_chunks, results_metadata):
        # Format context with source information
        formatted_chunks = []
        
        for i, (chunk, metadata) in enumerate(zip(relevant_chunks, results_metadata)):
            # Format with page information if available
            source_info = f"Source: {metadata.get('source', 'Unknown')}"
            if metadata.get('chunk_number'):
                source_info += f" (Chunk {metadata['chunk_number']})"
            
            formatted_chunk = f"{source_info}\n{chunk}"
            formatted_chunks.append(formatted_chunk)
            
        context = "\n\n---\n\n".join(formatted_chunks)  # Added separator for better readability
        
        rag_assistant = st.session_state.rag_assistant

        try:
            answer = asyncio.run(rag_assistant.generate_response(
                user_prompt=question,
                context=context
            ))
        except Exception as e:
            return f"Error generating response: {str(e)}"

        return answer
