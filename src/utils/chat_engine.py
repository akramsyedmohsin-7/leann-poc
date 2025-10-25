"""
RAG Chat Engine using LEANN Vector Database
"""

import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os


class RAGChatEngine:
    """Chat engine that uses LEANN for RAG"""

    def __init__(self, vector_db, model: str = "gpt-3.5-turbo"):
        """
        Initialize chat engine

        Args:
            vector_db: LeannVectorDB instance
            model: OpenAI model to use
        """
        self.vector_db = vector_db
        self.model = model
        self.client = None

        # Initialize OpenAI client if API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        question: str,
        top_k: int = 3,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG

        Args:
            question: User question
            top_k: Number of documents to retrieve
            temperature: LLM temperature

        Returns:
            Dictionary containing response and metrics
        """
        if not self.client:
            return {
                "response": "Error: OpenAI API key not configured",
                "error": "OPENAI_API_KEY not found in environment",
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": 0,
                "context_docs": [],
                "num_docs_used": 0
            }

        total_start = time.time()

        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        search_results = self.vector_db.search(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start

        context_docs = search_results.get('results', [])

        # Step 2: Build prompt with context
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(context_docs)
        ])

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the context documents to answer the user's question accurately.
If the context doesn't contain relevant information, say so clearly."""

        user_prompt = f"""Context:
{context_text}

Question: {question}

Please provide a detailed answer based on the context above."""

        # Step 3: Generate response
        generation_start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )

            answer = response.choices[0].message.content
            generation_time = time.time() - generation_start

            total_time = time.time() - total_start

            return {
                "response": answer,
                "context_docs": context_docs,
                "num_docs_used": len(context_docs),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "model": self.model,
                "search_time": search_results.get('search_time', 0)
            }

        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "error": str(e),
                "context_docs": context_docs,
                "num_docs_used": len(context_docs),
                "retrieval_time": retrieval_time,
                "generation_time": 0,
                "total_time": time.time() - total_start,
                "model": self.model
            }

    def is_available(self) -> bool:
        """Check if chat engine is available (OpenAI key configured)"""
        return self.client is not None
