# test_vdb_geojson

## Core Thesis

> **Why vector-embed GeoJSON instead of building better SQL / query tools for agents?**

Traditional approaches give LLM agents SQL tools or custom query APIs to retrieve geographic data. This project tests a different hypothesis: **embed each GeoJSON feature as a natural-language text document**, store the embeddings in a vector database, and let agents find relevant features through **semantic similarity search** — no SQL, no schema knowledge, no custom query tools required.

Benefits of this approach:
- Agents describe what they're looking for in plain English; the vector DB handles retrieval
- Semantic search captures intent that SQL filters miss (e.g., "large fires near populated areas")
- The same pipeline works for any GeoJSON dataset — no per-dataset query tooling
- Embeddings encode spatial + attribute context together in a unified representation

## Dataset

`sample_wildfire.json` — 656 wildfire incident points sourced from the NIFC/IRWIN data feed, containing attributes such as:
- `IncidentName`, `IncidentTypeCategory`, `UniqueFireIdentifier`
- `DailyAcres`, `CalculatedAcres`, `PercentContained`
- `FireDiscoveryDateTime`, `ICS209ReportDateTime`
- `POOState`, `POOCounty`, `FireCause`, `ComplexityLevel`
- Point geometry (longitude / latitude in WGS-84)

## Implementations

### Python (LangChain + ChromaDB + sentence-transformers)

Uses `all-MiniLM-L6-v2` (runs locally, no API key needed) to embed each
wildfire feature and stores vectors in ChromaDB.

```bash
cd python
pip install -r requirements.txt

# 1. Embed all 656 wildfire features and persist to ChromaDB
python embed_geojson.py

# 2. Run semantic queries (interactive or single-shot)
python query_geojson.py                          # interactive mode
python query_geojson.py --query "large fire in California" --k 5
```

### JavaScript (LangChain.js + ChromaDB + Xenova/transformers)

Uses the same `all-MiniLM-L6-v2` model via `@xenova/transformers` (runs
in-process, no API key needed) and stores vectors in a ChromaDB server.

**Prerequisites:** Start a local ChromaDB server before running:
```bash
pip install chromadb
chroma run --path ./chroma_data
```

Then, in a separate terminal:
```bash
cd javascript
npm install

# 1. Embed all 656 wildfire features
npm run embed

# 2. Run semantic queries
npm run query                             # interactive mode
node query_geojson.js "lightning fire"    # single query
```

## How It Works

```
sample_wildfire.json
       │
       ▼
 feature_to_text()          ← attributes + geometry → natural-language string
       │
       ▼
 HuggingFace Embeddings      ← all-MiniLM-L6-v2 (384-dim vectors)
       │
       ▼
 ChromaDB Vector Store       ← cosine similarity index
       │
       ▼
 similarity_search(query)    ← semantic search, no SQL needed
```

## Example Queries

```
"large wildfire in California"
"fire with low containment percentage"
"lightning caused fire in the Pacific Northwest"
"complex fire requiring federal resources"
"recently discovered fire under 100 acres"
```

## Extending This Project

- Swap `all-MiniLM-L6-v2` for `text-embedding-3-small` (OpenAI) for higher accuracy
- Add a LangChain retrieval QA chain to answer natural-language questions grounded in the data
- Try other GeoJSON datasets (flood zones, parcels, infrastructure) with zero query tool changes
- Benchmark semantic search recall vs. SQL attribute filtering
