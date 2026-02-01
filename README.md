# Athenaeum

A Python library that equips AI agents with tools for intelligent interaction with knowledge bases. Athenaeum handles document ingestion, semantic search, and structured content access, making it suitable for agent-based systems, RAG pipelines, and automation workflows.

## Installation

```bash
pip install athenaeum-kb
```

### Optional OCR backends

```bash
pip install athenaeum-kb[docling]   # Docling document converter
pip install athenaeum-kb[mistral]   # Mistral cloud OCR (PDF only)
pip install athenaeum-kb[lighton]   # LightOn local model (PDF + images)
pip install athenaeum-kb[all-ocr]   # All OCR backends
```

## Quick start

```python
from langchain_openai import OpenAIEmbeddings
from athenaeum import Athenaeum, AthenaeumConfig

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
kb = Athenaeum(embeddings=embeddings)

# Load a document
doc_id = kb.load_doc("report.pdf")

# Search across all documents
hits = kb.search_docs("quarterly revenue", top_k=5)

# Search within a specific document
chunks = kb.search_doc_contents(doc_id, "executive summary")

# Read specific lines
excerpt = kb.read_doc(doc_id, start_line=1, end_line=50)

# List all loaded documents
docs = kb.list_docs()
```

## Tools

### `load_doc`

Load a document into the knowledge base, automatically extracting content, metadata, and embeddings.

```python
load_doc(path: str) -> str
```

**Parameters:**
- `path`: Path to the document file

**Supported formats:** PDF, PPTX, DOCX, XLSX, JSON, CSV, TXT, MD, HTML, XML, RTF, EPUB

**Returns:** A document identifier (`doc_id`) for subsequent operations.

### `list_docs`

List all documents currently stored in the knowledge base.

```python
list_docs() -> list[SearchHit]
```

**Returns:** A list of documents with metadata (id, name, line count, table of contents) and relevance scores.

### `search_docs`

Search across all documents in the knowledge base.

```python
search_docs(
    query: str,
    top_k: int = 10,
    scope: Literal["names", "contents"] = "contents",
    strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
) -> list[SearchHit]
```

**Parameters:**
- `query`: Search query text
- `top_k`: Maximum number of results (default: 10)
- `scope`: Where to search
  - `"contents"`: Search within document contents (default)
  - `"names"`: Search only document names
- `strategy`: Search strategy (only applies when scope is `"contents"`)
  - `"hybrid"`: Combines vector and BM25 search (default)
  - `"bm25"`: Keyword-based search only
  - `"vector"`: Semantic similarity search only

**Returns:** A ranked list of `SearchHit` objects matching the query.

### `search_doc_contents`

Search within a specific document.

```python
search_doc_contents(
    doc_id: str,
    query: str,
    top_k: int = 5,
    strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
) -> list[ContentSearchHit]
```

**Parameters:**
- `doc_id`: Document identifier
- `query`: Search query text
- `top_k`: Maximum number of results (default: 5)
- `strategy`: Search strategy (`"hybrid"`, `"bm25"`, or `"vector"`)

**Returns:** A list of matching content fragments with line ranges and relevance scores.

### `read_doc`

Read a specific range of lines from a document.

```python
read_doc(
    doc_id: str,
    start_line: int = 1,
    end_line: int = 100,
) -> Excerpt
```

**Parameters:**
- `doc_id`: Document identifier
- `start_line`: Starting line number (1-indexed, default: 1)
- `end_line`: Ending line number (1-indexed, inclusive, default: 100)

**Returns:** An `Excerpt` containing the requested lines.

## Configuration

```python
from pathlib import Path
from athenaeum import AthenaeumConfig

config = AthenaeumConfig(
    storage_dir=Path.home() / ".athenaeum",  # Where to store documents and indexes
    chunk_size=80,                            # Lines per chunk
    chunk_overlap=20,                         # Overlapping lines between chunks
    rrf_k=60,                                 # RRF constant for hybrid search
    default_strategy="hybrid",                # Default search strategy
)

kb = Athenaeum(embeddings=embeddings, config=config)
```

## OCR backends

Athenaeum supports multiple document-to-markdown converters:

| Backend | Formats | Notes |
|---------|---------|-------|
| **markitdown** (default) | PDF, PPTX, DOCX, XLSX, JSON, CSV, TXT, MD, HTML, XML, RTF, EPUB | Included in base install |
| **docling** | PDF, PPTX, DOCX, XLSX, HTML, MD | `pip install athenaeum-kb[docling]` |
| **mistral** | PDF | Cloud API, requires `MISTRAL_API_KEY` |
| **lighton** | PDF, PNG, JPG, JPEG, TIFF, BMP | Local transformer model, supports GPU |

```python
from athenaeum import Athenaeum, get_ocr_provider

ocr = get_ocr_provider("docling")
kb = Athenaeum(embeddings=embeddings, ocr_provider=ocr)
```

### Custom OCR provider

```python
from athenaeum.ocr import CustomOCR
from pathlib import Path

def my_converter(file_path: Path) -> str:
    return "markdown content"

ocr = CustomOCR(fn=my_converter, extensions={".custom"})
kb = Athenaeum(embeddings=embeddings, ocr_provider=ocr)
```

## Search strategies

- **Hybrid** (default): Combines vector similarity and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Vector**: Semantic similarity search via embeddings (Chroma-backed).
- **BM25**: Traditional keyword-based ranking using the BM25Okapi algorithm.

## Document ingestion workflow

When `load_doc(path)` is called:

1. **Validation** -- verify the file exists and the format is supported.
2. **Content extraction** -- convert the file to Markdown using the configured OCR backend.
3. **Pre-processing** -- generate metadata, extract a table of contents from headings, and chunk the Markdown with heading-aware boundary snapping.
4. **Indexing** -- generate vector embeddings and store them in Chroma; add chunks to the BM25 index.

## Data models

| Model | Description |
|-------|-------------|
| `Document` | Full document record (id, name, paths, line count, TOC, timestamps) |
| `SearchHit` | Document-level search result with score and snippet |
| `ContentSearchHit` | Within-document search result with line range and text |
| `Excerpt` | Text fragment from `read_doc` |
| `TOCEntry` | Table of contents entry (title, level, line range) |
| `ChunkMetadata` | Internal chunk metadata for indexing |
| `Metadata` | Lightweight id + name pair |

## Development

```bash
pip install athenaeum-kb[dev]
pytest
ruff check src/
mypy src/
```

## License

MIT
