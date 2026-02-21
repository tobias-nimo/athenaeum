# Athenaeum

A Python library that equips AI agents with tools for intelligent interaction with knowledge bases. Athenaeum handles document ingestion, semantic search, and structured content access, making it suitable for agent-based systems, RAG pipelines, and automation workflows.

## Installation

```bash
pip install athenaeum-kb
```

### Optional OCR backends

```bash
pip install athenaeum-kb[docling]   # Docling document converter
pip install athenaeum-kb[mistral]   # Mistral cloud OCR (PDF + images)
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
load_doc(path: str, tags: set[str] | None = None) -> str
```

**Parameters:**
- `path`: Path to the document file
- `tags`: Optional set of tags to assign to the document

**Supported formats:** PDF, PPTX, DOCX, XLSX, JSON, CSV, TXT, MD, HTML, XML, RTF, EPUB

**Returns:** A document identifier (`doc_id`) for subsequent operations.

### `list_docs`

List all documents currently stored in the knowledge base.

```python
list_docs(tags: set[str] | None = None) -> list[SearchHit]
```

**Parameters:**
- `tags`: Optional set of tags to filter by (OR semantics)

**Returns:** A list of documents with metadata (id, name, line count, table of contents, tags) and relevance scores.

### `search_docs`

Search across all documents in the knowledge base.

```python
search_docs(
    query: str,
    top_k: int = 10,
    scope: Literal["names", "contents"] = "contents",
    strategy: Literal["hybrid", "bm25", "vector"] = "hybrid",
    tags: set[str] | None = None,
    aggregate: bool = True,
) -> list[SearchHit] | list[ContentSearchHit]
```

**Parameters:**
- `query`: Search query text
- `top_k`: Maximum number of results (default: 10)
- `tags`: Optional set of tags to filter by (OR semantics)
- `scope`: Where to search
  - `"contents"`: Search within document contents (default)
  - `"names"`: Search only document names
- `strategy`: Search strategy (only applies when scope is `"contents"`)
  - `"hybrid"`: Combines vector and BM25 search (default)
  - `"bm25"`: Keyword-based search only
  - `"vector"`: Semantic similarity search only
- `aggregate`: If `True` (default), collapse chunk results into one `SearchHit` per document. If `False`, return raw `ContentSearchHit` objects with exact line ranges.

**Returns:** When `aggregate=True`: a ranked list of `SearchHit` objects (one per document). When `aggregate=False`: a ranked list of `ContentSearchHit` objects (one per chunk).

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
    chunk_size=1500,                         # Characters per chunk
    chunk_overlap=200,                       # Overlapping characters between chunks
    rrf_k=60,                                # RRF constant for hybrid search
    default_strategy="hybrid",               # Default search strategy
    similarity_threshold=None,               # Min cosine score [0, 1]; None = no filter
)

kb = Athenaeum(embeddings=embeddings, config=config)
```

### Custom text splitter

Pass any object with a `split_text(str) -> list[str]` method to override the default chunking strategy. This is useful for token-based splitting or domain-specific separator rules.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import tiktoken

enc = tiktoken.get_encoding("cl100k_base") # or tiktoken.encoding_for_model("text-embedding-3-small")
token_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.MARKDOWN,
    chunk_size=256,
    chunk_overlap=32,
    length_function=lambda text: len(enc.encode(text)),
)

kb = Athenaeum(embeddings=embeddings, config=config, text_splitter=token_splitter)
```

> **Note:** When `text_splitter` is provided, `chunk_size` and `chunk_overlap` in `AthenaeumConfig` are ignored — sizing is controlled entirely by the splitter.

### Similarity threshold

Filter out low-confidence vector results by setting `similarity_threshold` in `AthenaeumConfig`. Scores are cosine similarity values in [0, 1]; only chunks that meet or exceed the threshold are returned.

```python
config = AthenaeumConfig(similarity_threshold=0.35)
kb = Athenaeum(embeddings=embeddings, config=config)

# Low-scoring chunks are silently dropped from vector and hybrid results
hits = kb.search_docs("quarterly revenue", strategy="vector")
```

> **Breaking change:** Athenaeum now creates Chroma collections with `hnsw:space=cosine`. If you have an existing persistent index (in `index/chroma/`) that was created without this setting, **delete the directory and re-index your documents** to get correct scores. Existing collections retain their original L2 distance function and will continue to produce scores outside [0, 1].

### Chunk-level results

Pass `aggregate=False` to `search_docs` to receive raw chunk-level hits instead of one result per document. Each hit includes the exact line range, making it easy to pinpoint where a match occurs.

```python
chunks = kb.search_docs("quarterly revenue", aggregate=False)
for hit in chunks:
    print(f"{hit.name} lines {hit.line_range[0]}-{hit.line_range[1]}: {hit.text[:80]}")
```

## OCR backends

Athenaeum supports multiple document-to-markdown converters:

| Backend | Formats | Notes |
|---------|---------|-------|
| **markitdown** (default) | PDF, PPTX, DOCX, XLSX, JSON, CSV, TXT, MD, HTML, XML, RTF, EPUB | Included in base install |
| **docling** | PDF, PPTX, DOCX, XLSX, HTML, MD | `pip install athenaeum-kb[docling]` |
| **mistral** | PDF, PNG, JPG, JPEG, AVIF | Cloud API, requires `MISTRAL_API_KEY` |
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

## Tags

Documents can be tagged with free-form labels for filtering. Tags use OR semantics: filtering by `{"a", "b"}` returns documents tagged with either `a` or `b` (or both).

```python
# Load with tags
doc_id = kb.load_doc("report.pdf", tags={"finance", "Q4"})

# Add/remove tags later
kb.tag_doc(doc_id, {"important"})
kb.untag_doc(doc_id, {"Q4"})

# List all tags in the knowledge base
all_tags = kb.list_tags()  # {"finance", "important"}

# Filter list_docs by tags
finance_docs = kb.list_docs(tags={"finance"})

# Filter search_docs by tags
hits = kb.search_docs("revenue", tags={"finance"})
```

### `tag_doc`

Add tags to an existing document.

```python
tag_doc(doc_id: str, tags: set[str]) -> None
```

### `untag_doc`

Remove tags from an existing document.

```python
untag_doc(doc_id: str, tags: set[str]) -> None
```

### `list_tags`

Return all tags across all documents.

```python
list_tags() -> set[str]
```

## Search strategies

- **Hybrid** (default): Combines vector similarity and BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Vector**: Semantic similarity search via embeddings (Chroma-backed).
- **BM25**: Traditional keyword-based ranking using the BM25Okapi algorithm.

## Document ingestion workflow

When `load_doc(path)` is called:

1. **Validation** -- verify the file exists and the format is supported.
2. **Content extraction** -- convert the file to Markdown using the configured OCR backend.
3. **Pre-processing** -- generate metadata, extract a table of contents from headings, and chunk the Markdown using `RecursiveCharacterTextSplitter` with markdown-aware separators (headings first, then paragraphs, then lines).
4. **Indexing** -- generate vector embeddings and store them in Chroma; add chunks to the BM25 index.

## Data models

| Model | Description |
|-------|-------------|
| `Document` | Full document record (id, name, paths, line count, TOC, timestamps) |
| `SearchHit` | Document-level search result with score and snippet |
| `ContentSearchHit` | Within-document search result with line range, text, and optional `name` (populated by `search_docs(aggregate=False)`) |
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
