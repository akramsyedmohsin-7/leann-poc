"""
Document Processing Utilities
Handles loading and processing various document types
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
import docx


class DocumentProcessor:
    """Process various document types and extract text"""

    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.md'}

    @staticmethod
    def process_file(file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Process a single file and extract text

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        extension = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        metadata = {
            "filename": file_path.name,
            "extension": extension,
            "size_bytes": file_size,
            "path": str(file_path)
        }

        if extension == '.txt' or extension == '.md':
            text = DocumentProcessor._process_text(file_path)
        elif extension == '.pdf':
            text = DocumentProcessor._process_pdf(file_path)
        elif extension == '.docx':
            text = DocumentProcessor._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())

        return text, metadata

    @staticmethod
    def _process_text(file_path: Path) -> str:
        """Process text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    @staticmethod
    def _process_pdf(file_path: Path) -> str:
        """Process PDF files"""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    @staticmethod
    def _process_docx(file_path: Path) -> str:
        """Process DOCX files"""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    @staticmethod
    def process_directory(directory_path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process all supported files in a directory

        Args:
            directory_path: Path to the directory

        Returns:
            Tuple of (list of texts, list of metadata dicts)
        """
        texts = []
        metadatas = []

        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS:
                try:
                    text, metadata = DocumentProcessor.process_file(file_path)
                    texts.append(text)
                    metadatas.append(metadata)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

        return texts, metadatas

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks
