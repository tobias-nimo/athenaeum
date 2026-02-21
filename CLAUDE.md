# Athenaeum

A Python library that equips AI agents with tools for intelligent knowledge base interaction. Handles document ingestion, semantic search, and structured content access.

## Quick Reference

```bash
# Install
pip install athenaeum-kb

# Dev setup
pip install athenaeum-kb[dev]

# Run tests
pytest

# Lint & type check
ruff check src/
mypy src/
```

## Project Structure

```
src/athenaeum/
├── athenaeum.py      # Main Athenaeum orchestrator class
├── config.py         # AthenaeumConfig dataclass
├── models.py         # Pydantic models (Document, SearchHit, Excerpt, etc.)
├── document_store.py # JSON-backed document registry
├── storage.py        # File system layout manager
├── chunker.py        # Markdown chunking via LangChain RecursiveCharacterTextSplitter
├── toc.py            # Table of contents extraction from markdown
├── ocr/              # Document-to-markdown converters
│   ├── base.py       # OCRProvider ABC
│   ├── markitdown.py # Default backend (most formats)
│   ├── docling.py    # Optional: pip install athenaeum-kb[docling]
│   ├── mistral.py    # Optional: pip install athenaeum-kb[mistral]
│   ├── lighton.py    # Optional: pip install athenaeum-kb[lighton]
│   └── custom.py     # User-supplied callable wrapper
└── search/
    ├── bm25.py       # BM25Okapi keyword search
    ├── vector.py     # Chroma-backed vector similarity search
    └── hybrid.py     # Reciprocal Rank Fusion (RRF)
```

## Architecture

### Core Flow

1. **load_doc(path)** → validate → OCR convert to markdown → extract TOC → chunk → index (BM25 + vector)
2. **search_docs(query)** → run search strategy → aggregate by document → return ranked results
3. **search_doc_contents(doc_id, query)** → search within specific document → return chunks
4. **read_doc(doc_id, start_line, end_line)** → read specific line range

### Storage Layout

```
~/.athenaeum/
├── docs/<doc_id>/
│   ├── raw.*        # Original file
│   └── content.md   # Converted markdown
├── index/chroma/    # Chroma persistent directory
└── metadata.json    # Document registry
```

### Data Models (models.py)

| Model | Purpose |
|-------|---------|
| `Document` | Full document record with paths, TOC, timestamps, tags |
| `SearchHit` | Document-level search result |
| `ContentSearchHit` | Within-document search result with line range; `name` populated by `search_docs(aggregate=False)` |
| `Excerpt` | Text fragment from read_doc |
| `TOCEntry` | Table of contents entry (title, level, line range) |
| `ChunkMetadata` | Internal chunk for indexing |

### Search Strategies

- **hybrid** (default): BM25 + vector combined via RRF (k=60)
- **bm25**: Keyword-based only (rank_bm25.BM25Okapi)
- **vector**: Semantic similarity via Chroma

### Tags

Documents support free-form tags with OR semantics for filtering:
```python
kb.load_doc("file.pdf", tags={"finance", "Q4"})
kb.tag_doc(doc_id, {"important"})
kb.list_docs(tags={"finance"})  # Filter by tag
```

## Testing

```python
# FakeEmbeddings for deterministic tests (see tests/test_athenaeum.py)
class FakeEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        # Deterministic 32-dim vector from text
        ...
```

Fixtures in `tests/fixtures/`: sample.md, sample.txt

## Configuration

```python
AthenaeumConfig(
    storage_dir=Path.home() / ".athenaeum",  # Storage root
    chunk_size=1500,                          # Characters per chunk
    chunk_overlap=200,                        # Overlap in characters between consecutive chunks
    rrf_k=60,                                 # RRF constant for hybrid search
    default_strategy="hybrid",                # Default search strategy
    similarity_threshold=None,               # Min cosine score [0,1]; None = no filter
)
```

## Key APIs

```python
from athenaeum import Athenaeum, AthenaeumConfig, get_ocr_provider

# Initialize
kb = Athenaeum(embeddings=embeddings, config=config, ocr_provider=ocr)
# With custom text splitter (e.g. token-based)
# from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4o")
# token_splitter = RecursiveCharacterTextSplitter.from_language(
#     Language.MARKDOWN, chunk_size=256, chunk_overlap=32,
#     length_function=lambda text: len(enc.encode(text)),
# )
# kb = Athenaeum(embeddings=embeddings, ocr_provider=ocr, config=config, text_splitter=token_splitter)

# Core methods
doc_id = kb.load_doc(path, tags=None)           # Load document
kb.list_docs(tags=None)                          # List all documents
kb.search_docs(query, top_k, scope, strategy, tags, aggregate=True)  # Search across docs
kb.search_doc_contents(doc_id, query, top_k, strategy)  # Search within doc
kb.read_doc(doc_id, start_line, end_line)        # Read line range

# Tag management
kb.tag_doc(doc_id, tags)
kb.untag_doc(doc_id, tags)
kb.list_tags()
```

## OCR Backends

| Backend | Formats | Notes |
|---------|---------|-------|
| markitdown | PDF, PPTX, DOCX, XLSX, JSON, CSV, TXT, MD, HTML, XML, RTF, EPUB | Default |
| docling | PDF, PPTX, DOCX, XLSX, HTML, MD | pip install athenaeum-kb[docling] |
| mistral | PDF, PNG, JPG, JPEG, AVIF | Cloud API, requires MISTRAL_API_KEY |
| lighton | PDF, PNG, JPG, JPEG, TIFF, BMP | Local transformer model |

## Development Notes

- Python 3.11+ required
- Uses Pydantic v2 for models
- LangChain for embeddings interface and Chroma integration
- Line numbers are 1-indexed in all APIs
- Chunking uses `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)` by default; pass a custom `text_splitter` to `Athenaeum()` for token-based or other strategies
- BM25 index rebuilds on startup from stored markdown files
- Vector index persists in Chroma directory
