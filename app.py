"""
LEANN Vector Database Testing App
Main Streamlit application for testing LEANN vector database
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LEANN Vector DB Test",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = []
if 'build_metrics' not in st.session_state:
    st.session_state.build_metrics = {}

# Main header
st.markdown('<div class="main-header">🔍 LEANN Vector Database Testing</div>', unsafe_allow_html=True)

st.markdown("""
Welcome to the **LEANN Vector Database Testing Application**!

LEANN is the world's smallest vector index, achieving **97% storage savings** compared to traditional vector databases.

### Features:
- 📁 Upload documents (PDF, DOCX, TXT, MD)
- 🔧 Build vector indexes with LEANN
- 📊 Analytics dashboard with performance metrics
- 🔍 Search and query capabilities
- 💬 RAG Chat interface (MVP2)
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Backend selection
    backend = st.selectbox(
        "Vector Backend",
        ["hnsw", "diskann"],
        help="HNSW: General use | DiskANN: Large-scale deployments"
    )

    # Langfuse status
    st.subheader("🔌 Integrations")
    langfuse_configured = all([
        os.getenv('LANGFUSE_PUBLIC_KEY'),
        os.getenv('LANGFUSE_SECRET_KEY')
    ])

    if langfuse_configured:
        st.success("✅ Langfuse: Connected")
    else:
        st.warning("⚠️ Langfuse: Not configured")
        st.info("Add credentials to .env to enable tracking")

    st.divider()

    # Navigation
    st.header("📍 Navigation")
    page = st.radio(
        "Go to:",
        ["📁 Upload & Index", "📊 Analytics", "🔍 Search", "💬 Chat (MVP2)"],
        label_visibility="collapsed"
    )

# Main content based on navigation
if page == "📁 Upload & Index":
    from utils import DocumentProcessor, LangfuseTracker
    from leann import LeannVectorDB
    import time

    st.header("📁 Upload Documents & Build Vector Index")

    # File upload
    st.subheader("1️⃣ Upload Documents")

    upload_method = st.radio(
        "Choose upload method:",
        ["Upload Files", "Upload Folder"]
    )

    uploaded_files = None
    folder_path = None

    if upload_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, MD"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

            # Save uploaded files
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

    else:
        folder_path = st.text_input(
            "Enter folder path:",
            placeholder="/path/to/your/documents"
        )

        if folder_path and Path(folder_path).exists():
            st.success(f"✅ Folder found: {folder_path}")
        elif folder_path:
            st.error("❌ Folder not found")

    st.divider()

    # Build vector index
    st.subheader("2️⃣ Build Vector Index")

    col1, col2 = st.columns([3, 1])

    with col1:
        index_name = st.text_input(
            "Index Name",
            value="my_index",
            help="Name for your vector index"
        )

    with col2:
        st.write("")
        st.write("")
        build_button = st.button("🚀 Create Vector Index", type="primary", use_container_width=True)

    if build_button:
        if not uploaded_files and not (folder_path and Path(folder_path).exists()):
            st.error("⚠️ Please upload files or specify a valid folder path first!")
        else:
            # Initialize tracker
            tracker = LangfuseTracker()

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Process documents
                status_text.text("📖 Processing documents...")
                progress_bar.progress(20)

                texts = []
                metadatas = []

                if upload_method == "Upload Files":
                    upload_dir = Path("data/uploads")
                    for file_path in upload_dir.glob('*'):
                        if file_path.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS:
                            text, metadata = DocumentProcessor.process_file(file_path)
                            texts.append(text)
                            metadatas.append(metadata)
                else:
                    texts, metadatas = DocumentProcessor.process_directory(Path(folder_path))

                st.session_state.uploaded_files_data = metadatas

                # Step 2: Initialize vector DB
                status_text.text("🔧 Initializing LEANN vector database...")
                progress_bar.progress(40)

                index_path = Path("data/vectors") / index_name
                vector_db = LeannVectorDB(str(index_path), backend=backend)

                # Step 3: Add documents
                status_text.text(f"📝 Adding {len(texts)} documents...")
                progress_bar.progress(60)

                start_time = time.time()
                add_metrics = vector_db.add_documents(texts, metadatas)
                add_time = time.time() - start_time

                # Track with Langfuse
                tracker.track_indexing("add_documents", add_metrics, {"backend": backend})

                # Step 4: Build index
                status_text.text("🏗️ Building vector index...")
                progress_bar.progress(80)

                build_metrics = vector_db.build_index()

                # Track with Langfuse
                tracker.track_indexing("build_index", build_metrics, {"backend": backend})

                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Vector index created successfully!")

                # Store in session state
                st.session_state.vector_db = vector_db
                st.session_state.index_built = True
                st.session_state.build_metrics = {
                    **add_metrics,
                    **build_metrics,
                    "backend": backend
                }

                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### 🎉 Success! Vector Index Created")
                st.markdown("</div>", unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Documents", build_metrics['num_documents'])

                with col2:
                    st.metric("Build Time", f"{build_metrics['build_time']:.2f}s")

                with col3:
                    st.metric(
                        "Storage Savings",
                        f"{build_metrics['storage_savings_percent']:.1f}%"
                    )

                with col4:
                    st.metric(
                        "Index Size",
                        LeannVectorDB.format_size(build_metrics['index_size'])
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")

    if not st.session_state.index_built:
        st.warning("⚠️ No index built yet. Please go to 'Upload & Index' first.")
    else:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd

        metrics = st.session_state.build_metrics
        files_data = st.session_state.uploaded_files_data

        # Overview metrics
        st.subheader("📈 Overview Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Documents",
                metrics.get('num_documents', 0)
            )

        with col2:
            st.metric(
                "Total Build Time",
                f"{metrics.get('build_time', 0):.2f}s"
            )

        with col3:
            st.metric(
                "Original Size",
                LeannVectorDB.format_size(metrics.get('original_size', 0))
            )

        with col4:
            st.metric(
                "Index Size",
                LeannVectorDB.format_size(metrics.get('index_size', 0))
            )

        st.divider()

        # Storage comparison chart
        st.subheader("💾 Storage Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Index Size', 'Storage Saved'],
                values=[
                    metrics.get('index_size', 0),
                    metrics.get('original_size', 0) - metrics.get('index_size', 0)
                ],
                hole=.3,
                marker_colors=['#1f77b4', '#2ca02c']
            )])
            fig_pie.update_layout(
                title="Storage Efficiency",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    name='Original',
                    x=['Size'],
                    y=[metrics.get('original_size', 0)],
                    marker_color='#ff7f0e'
                ),
                go.Bar(
                    name='Compressed',
                    x=['Size'],
                    y=[metrics.get('index_size', 0)],
                    marker_color='#1f77b4'
                )
            ])
            fig_bar.update_layout(
                title="Size Comparison (Bytes)",
                barmode='group',
                height=400,
                yaxis_title="Bytes"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Document details
        st.subheader("📄 Document Details")

        if files_data:
            df = pd.DataFrame(files_data)
            st.dataframe(
                df[['filename', 'extension', 'size_bytes', 'char_count', 'word_count']],
                use_container_width=True
            )

            # Document size distribution
            fig_dist = px.bar(
                df,
                x='filename',
                y='size_bytes',
                title="Document Size Distribution",
                labels={'size_bytes': 'Size (Bytes)', 'filename': 'Document'}
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)

        # Performance metrics
        st.divider()
        st.subheader("⚡ Performance Metrics")

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.metric(
                "Avg Chars/Doc",
                f"{metrics.get('avg_chars_per_doc', 0):.0f}"
            )

        with perf_col2:
            docs_per_sec = metrics.get('num_documents', 0) / max(metrics.get('build_time', 1), 0.01)
            st.metric(
                "Docs/Second",
                f"{docs_per_sec:.2f}"
            )

        with perf_col3:
            st.metric(
                "Compression Ratio",
                f"{metrics.get('storage_savings_percent', 0):.1f}%"
            )

elif page == "🔍 Search":
    st.header("🔍 Search Vector Database")

    if not st.session_state.index_built or st.session_state.vector_db is None:
        st.warning("⚠️ No index built yet. Please go to 'Upload & Index' first.")
    else:
        from utils import LangfuseTracker
        import time

        st.markdown("""
        Search your vector database using natural language queries.
        LEANN will find the most relevant documents based on semantic similarity.
        """)

        # Search interface
        col1, col2 = st.columns([4, 1])

        with col1:
            query = st.text_input(
                "Enter your search query:",
                placeholder="What are you looking for?",
                label_visibility="collapsed"
            )

        with col2:
            top_k = st.number_input(
                "Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of results to return"
            )

        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

        if search_button and query:
            tracker = LangfuseTracker()

            with st.spinner("Searching..."):
                try:
                    # Perform search
                    search_results = st.session_state.vector_db.search(query, top_k=top_k)

                    # Track with Langfuse
                    tracker.track_search(query, search_results)

                    # Display results
                    st.success(f"✅ Found {search_results['num_results']} results in {search_results['search_time']:.3f}s")

                    if search_results['results']:
                        st.subheader("📝 Search Results")

                        for i, result in enumerate(search_results['results'], 1):
                            with st.expander(f"Result {i}", expanded=(i == 1)):
                                st.markdown(result)

                    else:
                        st.info("No results found. Try a different query.")

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Search Time", f"{search_results['search_time']:.3f}s")
                    with col2:
                        st.metric("Results Found", search_results['num_results'])
                    with col3:
                        st.metric("Top K", top_k)

                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")

        elif search_button:
            st.warning("Please enter a search query")

elif page == "💬 Chat (MVP2)":
    st.header("💬 RAG Chat Interface")

    if not st.session_state.index_built or st.session_state.vector_db is None:
        st.warning("⚠️ No index built yet. Please go to 'Upload & Index' first.")
    else:
        from utils import RAGChatEngine, LangfuseTracker
        import os

        # Check OpenAI configuration
        openai_configured = os.getenv('OPENAI_API_KEY') is not None

        if not openai_configured:
            st.error("❌ OpenAI API Key not configured")
            st.info("Add your OPENAI_API_KEY to the .env file to enable chat functionality")
            st.code("""
# Add to .env file:
OPENAI_API_KEY=your_api_key_here
            """)
        else:
            st.success("✅ Chat engine ready")

            # Initialize chat engine
            if 'chat_engine' not in st.session_state:
                st.session_state.chat_engine = RAGChatEngine(st.session_state.vector_db)

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Settings
            with st.expander("⚙️ Chat Settings"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    model = st.selectbox(
                        "Model",
                        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                        help="OpenAI model to use"
                    )
                    st.session_state.chat_engine.model = model

                with col2:
                    context_docs = st.number_input(
                        "Context Docs",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Number of documents to retrieve"
                    )

                with col3:
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Higher = more creative, Lower = more focused"
                    )

            st.divider()

            # Chat interface
            st.subheader("💬 Chat with Your Documents")

            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat['question'])

                with st.chat_message("assistant"):
                    st.write(chat['response'])

                    # Show metrics in expander
                    with st.expander("📊 Response Metrics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Time", f"{chat.get('total_time', 0):.2f}s")
                        with col2:
                            st.metric("Retrieval", f"{chat.get('retrieval_time', 0):.3f}s")
                        with col3:
                            st.metric("Generation", f"{chat.get('generation_time', 0):.2f}s")

                        # Show context documents
                        if chat.get('context_docs'):
                            st.write("**Context Documents Used:**")
                            for j, doc in enumerate(chat['context_docs'], 1):
                                with st.expander(f"Document {j}"):
                                    st.text(doc[:500] + "..." if len(doc) > 500 else doc)

            # Chat input
            question = st.chat_input("Ask a question about your documents...")

            if question:
                # Add user message
                with st.chat_message("user"):
                    st.write(question)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        tracker = LangfuseTracker()

                        # Get response
                        result = st.session_state.chat_engine.chat(
                            question,
                            top_k=context_docs,
                            temperature=temperature
                        )

                        # Track with Langfuse
                        if 'error' not in result:
                            tracker.track_chat(
                                question,
                                result['response'],
                                result['context_docs'],
                                result
                            )

                        # Display response
                        st.write(result['response'])

                        # Show metrics
                        if 'error' not in result:
                            with st.expander("📊 Response Metrics", expanded=False):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Time", f"{result['total_time']:.2f}s")
                                with col2:
                                    st.metric("Retrieval", f"{result['retrieval_time']:.3f}s")
                                with col3:
                                    st.metric("Generation", f"{result['generation_time']:.2f}s")
                                with col4:
                                    st.metric("Docs Used", result['num_docs_used'])

                                # Show context documents
                                st.write("**Context Documents Used:**")
                                for j, doc in enumerate(result['context_docs'], 1):
                                    with st.expander(f"Document {j}"):
                                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)

                        # Add to history
                        st.session_state.chat_history.append({
                            'question': question,
                            'response': result['response'],
                            'context_docs': result.get('context_docs', []),
                            'total_time': result.get('total_time', 0),
                            'retrieval_time': result.get('retrieval_time', 0),
                            'generation_time': result.get('generation_time', 0),
                            'num_docs_used': result.get('num_docs_used', 0)
                        })

            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    Built with ❤️ using LEANN Vector Database |
    Powered by Streamlit |
    Tracked with Langfuse
</div>
""", unsafe_allow_html=True)
