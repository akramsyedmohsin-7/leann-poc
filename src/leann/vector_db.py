"""
LEANN Vector Database Wrapper
Provides a clean interface for working with LEANN vector database
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from leann import LeannBuilder, LeannSearcher


class LeannVectorDB:
    """Wrapper class for LEANN vector database operations"""

    def __init__(self, index_path: str, backend: str = "hnsw"):
        """
        Initialize LEANN vector database

        Args:
            index_path: Path to store/load the index
            backend: Backend to use ('hnsw' or 'diskann')
        """
        self.index_path = Path(index_path)
        self.backend = backend
        self.builder = None
        self.searcher = None
        self.metadata = {
            "created_at": None,
            "backend": backend,
            "num_documents": 0,
            "total_text_size": 0,
            "index_size": 0,
            "build_time": 0,
            "documents": []
        }

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Add documents to the vector database

        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries for each document

        Returns:
            Dictionary containing metrics about the operation
        """
        start_time = time.time()

        # Initialize builder if not already done
        if self.builder is None:
            self.builder = LeannBuilder(backend_name=self.backend)

        # Track metrics
        total_chars = 0

        # Add each text document
        for i, text in enumerate(texts):
            self.builder.add_text(text)
            total_chars += len(text)

            # Store document metadata
            doc_meta = {
                "index": i,
                "size": len(text),
                "added_at": datetime.now().isoformat()
            }
            if metadatas and i < len(metadatas):
                doc_meta.update(metadatas[i])

            self.metadata["documents"].append(doc_meta)

        self.metadata["num_documents"] += len(texts)
        self.metadata["total_text_size"] += total_chars

        elapsed_time = time.time() - start_time

        return {
            "num_documents_added": len(texts),
            "total_chars": total_chars,
            "time_taken": elapsed_time,
            "avg_chars_per_doc": total_chars / len(texts) if texts else 0
        }

    def build_index(self) -> Dict[str, Any]:
        """
        Build and save the vector index

        Returns:
            Dictionary containing metrics about the build operation
        """
        if self.builder is None:
            raise ValueError("No documents added. Call add_documents() first.")

        start_time = time.time()

        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the index
        self.builder.build_index(str(self.index_path))

        build_time = time.time() - start_time

        # Calculate index size
        index_size = self._calculate_index_size()

        # Update metadata
        self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["build_time"] = build_time
        self.metadata["index_size"] = index_size

        # Save metadata
        self._save_metadata()

        # Calculate storage savings
        storage_savings = 0
        if self.metadata["total_text_size"] > 0:
            storage_savings = (1 - (index_size / self.metadata["total_text_size"])) * 100

        return {
            "build_time": build_time,
            "index_size": index_size,
            "original_size": self.metadata["total_text_size"],
            "storage_savings_percent": storage_savings,
            "num_documents": self.metadata["num_documents"]
        }

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search the vector database

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            Dictionary containing search results and metrics
        """
        if self.searcher is None:
            if not self.index_path.exists():
                raise ValueError(f"Index not found at {self.index_path}. Build index first.")
            self.searcher = LeannSearcher(str(self.index_path))

        start_time = time.time()
        results = self.searcher.search(query, top_k=top_k)
        search_time = time.time() - start_time

        return {
            "results": results,
            "search_time": search_time,
            "num_results": len(results) if results else 0,
            "query": query,
            "top_k": top_k
        }

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index

        Returns:
            Dictionary containing index metadata and statistics
        """
        if self.index_path.exists():
            self._load_metadata()

        return {
            "exists": self.index_path.exists(),
            "path": str(self.index_path),
            "metadata": self.metadata
        }

    def _calculate_index_size(self) -> int:
        """Calculate total size of index files in bytes"""
        total_size = 0
        if self.index_path.exists():
            if self.index_path.is_file():
                total_size = self.index_path.stat().st_size
            elif self.index_path.is_dir():
                for file_path in self.index_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        return total_size

    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = self.index_path.parent / f"{self.index_path.name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _load_metadata(self):
        """Load metadata from JSON file"""
        metadata_path = self.index_path.parent / f"{self.index_path.name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format bytes to human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
