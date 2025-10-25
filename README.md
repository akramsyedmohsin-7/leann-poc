# LEANN Vector Database Testing App

A comprehensive Streamlit application for testing and evaluating the LEANN vector database - the world's smallest vector index with 97% storage savings.

## Features

### MVP1 - Core Functionality
- **Document Upload & Processing**: Upload PDF, DOCX, TXT, and MD files
- **Vector Index Creation**: Build efficient vector indexes using LEANN
- **Performance Metrics**: Track build time, storage savings, and compression ratios
- **Analytics Dashboard**: Visualize performance with interactive charts
- **Search Interface**: Semantic search with similarity matching
- **Langfuse Integration**: Track and monitor all operations

### MVP2 - RAG Chat
- **Conversational AI**: Chat with your documents using OpenAI models
- **Context-Aware Responses**: Retrieve relevant documents for accurate answers
- **Performance Tracking**: Monitor retrieval and generation times
- **Chat History**: Keep track of your conversations
- **Accuracy Metrics**: Evaluate RAG performance

## Project Structure

```
leann-poc/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
├── src/
│   ├── leann/
│   │   ├── __init__.py
│   │   └── vector_db.py       # LEANN wrapper class
│   └── utils/
│       ├── __init__.py
│       ├── document_processor.py  # Document loading utilities
│       ├── langfuse_tracker.py    # Langfuse integration
│       └── chat_engine.py         # RAG chat engine
├── data/
│   ├── uploads/               # Uploaded documents
│   └── vectors/               # Vector indexes
└── pages/                     # Additional Streamlit pages
```

## Installation

### Prerequisites
- Python >= 3.9
- pip or uv package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd leann-poc
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your credentials:
   ```env
   # Langfuse (Optional - for metrics tracking)
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com

   # OpenAI (Required for MVP2 Chat)
   OPENAI_API_KEY=your_openai_api_key

   # Embedding Configuration
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload & Index

1. **Navigate to "Upload & Index"** page
2. **Choose upload method**:
   - Upload individual files, or
   - Specify a folder path
3. **Configure settings**:
   - Select backend (HNSW or DiskANN)
   - Set index name
4. **Click "Create Vector Index"**
5. **View metrics**:
   - Number of documents processed
   - Build time
   - Storage savings percentage
   - Index size

### 2. Analytics Dashboard

View comprehensive analytics about your vector database:

- **Overview Metrics**: Documents, build time, sizes
- **Storage Comparison**: Visual charts showing compression
- **Document Details**: Individual file statistics
- **Performance Metrics**: Processing speed and efficiency

### 3. Search

1. **Enter a search query** in natural language
2. **Set the number of results** (top_k)
3. **Click "Search"**
4. **View results** with:
   - Matched documents
   - Search time
   - Relevance scores

### 4. Chat (MVP2)

1. **Configure OpenAI API key** in `.env`
2. **Navigate to "Chat (MVP2)"**
3. **Configure chat settings**:
   - Model selection (GPT-3.5, GPT-4)
   - Number of context documents
   - Temperature
4. **Ask questions** about your documents
5. **View**:
   - AI-generated responses
   - Context documents used
   - Performance metrics
   - Chat history

## LEANN Vector Database

### What is LEANN?

LEANN (Low-storage Efficient Approximate Nearest Neighbor) is a revolutionary vector database that:

- **Saves 97% storage** compared to traditional vector databases
- **Maintains high accuracy** with 90% top-3 recall
- **Runs on personal devices** - optimized for laptops
- **Uses graph-based selective recomputation** for efficiency

### Backends

- **HNSW**: Hierarchical Navigable Small World - general purpose
- **DiskANN**: Optimized for large-scale deployments

### Performance

- **Index Size**: < 5% of original data
- **Query Speed**: < 2 seconds for most queries
- **Recall**: 90%+ top-3 accuracy

## Langfuse Integration

Track and monitor all operations:

- **Indexing operations**: Document addition and index building
- **Search queries**: Query performance and results
- **Chat interactions**: RAG pipeline monitoring
- **Performance metrics**: All timing and accuracy data

Get your Langfuse credentials at: https://cloud.langfuse.com

## Development

### Adding New Features

1. **Create new utility modules** in `src/utils/`
2. **Add new pages** in `pages/` directory
3. **Update `app.py`** for new navigation items
4. **Add dependencies** to `requirements.txt`

### Testing

Test the application with various document types:

```bash
# Create test documents
mkdir test_docs
echo "Sample document content" > test_docs/sample.txt

# Run the app and upload test_docs folder
streamlit run app.py
```

## Troubleshooting

### Common Issues

1. **Langfuse not connecting**
   - Verify credentials in `.env`
   - Check internet connection
   - The app works without Langfuse (tracking disabled)

2. **OpenAI errors**
   - Verify API key in `.env`
   - Check account credits
   - MVP1 features work without OpenAI

3. **File upload errors**
   - Check file format (PDF, DOCX, TXT, MD)
   - Verify file is not corrupted
   - Try smaller files first

4. **Index build failures**
   - Check disk space
   - Verify write permissions
   - Try with smaller dataset first

### Performance Tips

- Start with smaller datasets (< 100 documents)
- Use HNSW backend for faster builds
- Chunk large documents for better results
- Monitor RAM usage with large datasets

## API Reference

### LeannVectorDB

```python
from leann import LeannVectorDB

# Initialize
db = LeannVectorDB(index_path="path/to/index", backend="hnsw")

# Add documents
db.add_documents(texts=["doc1", "doc2"], metadatas=[...])

# Build index
metrics = db.build_index()

# Search
results = db.search(query="search query", top_k=5)

# Get info
info = db.get_index_info()
```

### DocumentProcessor

```python
from utils import DocumentProcessor

# Process single file
text, metadata = DocumentProcessor.process_file(Path("file.pdf"))

# Process directory
texts, metadatas = DocumentProcessor.process_directory(Path("docs/"))

# Chunk text
chunks = DocumentProcessor.chunk_text(text, chunk_size=500, overlap=50)
```

### RAGChatEngine

```python
from utils import RAGChatEngine

# Initialize
chat = RAGChatEngine(vector_db, model="gpt-3.5-turbo")

# Chat
result = chat.chat(
    question="What is this about?",
    top_k=3,
    temperature=0.7
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **LEANN**: UC Berkeley, CUHK, AWS, UC Davis researchers
- **Streamlit**: For the amazing web framework
- **Langfuse**: For observability and tracking
- **OpenAI**: For LLM capabilities

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review the troubleshooting section

## Roadmap

### Future Enhancements
- [ ] Batch document processing
- [ ] Custom embedding models
- [ ] Export/import indexes
- [ ] Advanced search filters
- [ ] Multi-user support
- [ ] API endpoints
- [ ] Docker deployment
- [ ] Cloud storage integration

---

**Built with ❤️ using LEANN Vector Database**
