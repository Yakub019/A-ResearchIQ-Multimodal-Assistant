# ResearchIQ 📚
### An Image-Aware Retrieval-Augmented Generation (RAG) System for Academic Research Papers

ResearchIQ is an AI-powered research assistant designed to help users understand academic research papers through natural-language question answering.

Unlike a traditional text-only RAG application, ResearchIQ also extracts descriptions of figures, charts, and diagrams from research papers, allowing the retrieval pipeline to use both textual content and visual information.

## 🚀 Project Overview

Research papers often contain important information in:
- Textual explanations
- Figures and diagrams
- Charts and visual representations
- Tables and technical descriptions

Traditional text-based document question-answering systems may miss important visual information.

ResearchIQ addresses this limitation by combining document processing, image-description extraction, vector search, and Large Language Models to provide context-aware answers with source citations.

## ✨ Features Implemented

### 1. PDF Document Ingestion
- Loads academic research papers in PDF format.
- Uses Docling for document conversion and processing.
- Uses LangChain Docling integration.
- Uses HybridChunker to split documents into meaningful chunks.
- Extracts document content and metadata.

### 2. Image-Aware Document Processing
- Enables picture-description extraction through Docling.
- Generates textual descriptions of figures and diagrams.
- Includes visual descriptions in document chunks.
- Makes visual information available to the downstream retrieval pipeline.

### 3. Embedding Generation
- Uses HuggingFace embeddings.
- Embedding model: `BAAI/bge-small-en-v1.5`.
- Generates 384-dimensional vector representations.
- Uses normalized embeddings.

### 4. Vector Database
- Uses ChromaDB for persistent vector storage.
- Stores research-paper chunks and their metadata.
- Uses deterministic document IDs.
- Maintains a separate image-aware collection: `researchiq_papers_image_aware`.

### 5. Semantic Retrieval
- Retrieves relevant document chunks based on user questions.
- Uses similarity search.
- Retrieves the top 4 chunks by default.
- Preserves source and page metadata for citations.

### 6. Question Answering
- Uses LangChain prompt templates and chains.
- Uses Groq as the LLM provider.
- Model: `llama-3.3-70b-versatile`.
- Generates answers using retrieved context.
- Supports questions about both textual content and extracted image descriptions.
- Includes source and page citations in answers.

### 7. Streamlit User Interface
- Interactive chat interface.
- Displays conversation history.
- Provides retrieved-source information.
- Displays source filenames and page numbers.
- Includes a clear-chat option.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Document Processing | Docling |
| Document Integration | LangChain Docling |
| Chunking | HybridChunker |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| LLM Provider | Groq |
| LLM | Llama 3.3 70B Versatile |
| User Interface | Streamlit |
| Package Management | uv |

## 🏗️ Current Architecture

```text
                  Research Papers (PDF)
                           |
                           v
                    Docling Loader
                           |
                           v
                     HybridChunker
                           |
                 +---------+---------+
                 |                   |
                 v                   v
           Text Extraction     Image Description
                 |                   |
                 +---------+---------+
                           |
                           v
                    Document Chunks
                           |
                           v
                  HuggingFace Embeddings
                           |
                           v
                       ChromaDB
                           |
                           v
                    User Question
                           |
                           v
                  Semantic Retriever
                           |
                           v
                   Retrieved Context
                           |
                           v
                    LangChain QA Chain
                           |
                           v
                      Groq LLM
                           |
                           v
                  Grounded Answer
                           |
                           v
                   Source Citations
                           |
                           v
                    Streamlit UI
```

## 📂 Project Structure

```text
A-ResearchIQ/
│
├── app/
│   ├── ingestion/
│   │   └── pdf_loader.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── vectorstore/
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── prompts/
│   │   └── qa_prompt.py
│   │
│   └── chains/
│       └── qa_chain.py
│
├── data/
│   └── papers/
│       ├── 01_sample.pdf
│       ├── 02_sample.pdf
│       └── ...
│
├── evals/
│
├── goldens/
│
├── chroma_db/
│
├── streamlit_app.py
├── pyproject.toml
├── uv.lock
├── .env
└── README.md
```

*Note: `chroma_db/`, `.env`, and `.venv/` are local runtime/environment files and should not be committed to Git. The structure above describes the development environment.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or later (use a Python version supported by your installed dependencies).
- Git
- uv package manager
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ResearchIQ.git
cd ResearchIQ
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Keep your API key private. Never commit `.env` to GitHub.

### 4. Add Research Papers

Place your PDF research papers inside:

```text
data/papers/
```

### 5. Process Research Papers

```bash
uv run python -m app.ingestion.pdf_loader
```

### 6. Create the Vector Database

```bash
uv run python -m app.vectorstore.vector_store
```

This processes the research papers and indexes their extracted content and image descriptions in ChromaDB.

### 7. Launch ResearchIQ

```bash
uv run streamlit run streamlit_app.py
```

The Streamlit application will provide an interactive interface for asking questions about the research papers.

## 💬 Example Questions

Try asking:

1. Explain the diagram showing how human plans are inferred.
2. What is the main contribution of this research paper?
3. Explain the important figures and their findings.
4. What are the key concepts discussed in the paper?
5. Compare the visual information with the textual explanation.

## 📌 Current Development Status

| Module | Status |
|---|---|
| PDF ingestion using Docling | Completed |
| Document chunking | Completed |
| Image-description extraction | Implemented |
| Embedding generation | Completed |
| ChromaDB indexing | Implemented |
| Semantic retrieval | Implemented |
| Image-aware retrieval | Implemented |
| LangChain QA chain | Implemented |
| Source citations | Implemented |
| Streamlit interface | Implemented |
| Evaluation integration | Initial integration |
| Advanced LangGraph workflow | Planned |

## 🔮 Future Enhancements

The following capabilities are planned for future development:

- Advanced multimodal retrieval.
- Improved figure and chart understanding.
- LangGraph-based agentic workflows.
- Query transformation and query expansion.
- Contextual retrieval and reranking.
- Citation verification.
- Answer evaluation and correction.
- Multi-paper comparison and summarization.
- Advanced research-paper insights.

## 🎯 Project Goal

The goal of ResearchIQ is to make academic research more accessible by enabling users to interact with research papers conversationally and retrieve meaningful insights from both textual and visual information.

---

**ResearchIQ — Making Research Papers Easier to Understand.**