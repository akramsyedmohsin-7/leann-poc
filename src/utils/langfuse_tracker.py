"""
Langfuse Integration for tracking metrics and observability
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from langfuse import Langfuse


class LangfuseTracker:
    """Track LEANN operations with Langfuse"""

    def __init__(self):
        """Initialize Langfuse client"""
        self.client = None
        self.enabled = False

        # Check if Langfuse credentials are available
        if all([
            os.getenv('LANGFUSE_PUBLIC_KEY'),
            os.getenv('LANGFUSE_SECRET_KEY'),
            os.getenv('LANGFUSE_HOST')
        ]):
            try:
                self.client = Langfuse(
                    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
                    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                self.enabled = True
            except Exception as e:
                print(f"Warning: Failed to initialize Langfuse: {str(e)}")
                self.enabled = False

    def track_indexing(
        self,
        operation: str,
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Track indexing operations

        Args:
            operation: Operation name (e.g., 'add_documents', 'build_index')
            metrics: Dictionary of metrics to track
            metadata: Additional metadata

        Returns:
            Trace ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            trace = self.client.trace(
                name=f"leann_{operation}",
                metadata=metadata or {},
                timestamp=datetime.now()
            )

            # Add metrics as a generation
            trace.generation(
                name=operation,
                model="leann-vector-db",
                metadata={
                    "operation": operation,
                    **metrics
                }
            )

            trace.update(
                output=metrics
            )

            return trace.id

        except Exception as e:
            print(f"Warning: Failed to track with Langfuse: {str(e)}")
            return None

    def track_search(
        self,
        query: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Track search operations

        Args:
            query: Search query
            results: Search results and metrics
            metadata: Additional metadata

        Returns:
            Trace ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            trace = self.client.trace(
                name="leann_search",
                input=query,
                metadata=metadata or {},
                timestamp=datetime.now()
            )

            trace.generation(
                name="search",
                model="leann-vector-db",
                input=query,
                output=results.get('results', []),
                metadata={
                    "search_time": results.get('search_time'),
                    "num_results": results.get('num_results'),
                    "top_k": results.get('top_k')
                }
            )

            return trace.id

        except Exception as e:
            print(f"Warning: Failed to track with Langfuse: {str(e)}")
            return None

    def track_chat(
        self,
        question: str,
        response: str,
        context_docs: List[str],
        metrics: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Track chat operations (for MVP2)

        Args:
            question: User question
            response: Generated response
            context_docs: Retrieved context documents
            metrics: Performance metrics
            metadata: Additional metadata

        Returns:
            Trace ID if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            trace = self.client.trace(
                name="leann_rag_chat",
                input=question,
                output=response,
                metadata=metadata or {},
                timestamp=datetime.now()
            )

            # Track retrieval
            trace.span(
                name="retrieval",
                input=question,
                output=context_docs,
                metadata={
                    "num_docs": len(context_docs),
                    "retrieval_time": metrics.get('retrieval_time')
                }
            )

            # Track generation
            trace.generation(
                name="generation",
                model=metrics.get('model', 'unknown'),
                input=question,
                output=response,
                metadata={
                    "generation_time": metrics.get('generation_time'),
                    "total_time": metrics.get('total_time')
                }
            )

            return trace.id

        except Exception as e:
            print(f"Warning: Failed to track with Langfuse: {str(e)}")
            return None

    def is_enabled(self) -> bool:
        """Check if Langfuse tracking is enabled"""
        return self.enabled
