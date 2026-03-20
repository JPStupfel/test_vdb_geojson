"""
embed_geojson.py

Demonstrates the core thesis: vector embedding GeoJSON features enables
semantic search for LLM agents — no SQL or custom query tools needed.

Each wildfire incident is converted to a natural-language text document,
embedded with a local sentence-transformer model, and stored in ChromaDB.
"""

import json
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
GEOJSON_PATH = REPO_ROOT / "sample_wildfire.json"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


def feature_to_text(feature: dict) -> str:
    """Convert a single GeoJSON feature into a natural-language description."""
    attrs = feature.get("attributes", {})
    geometry = feature.get("geometry", {})

    parts = []

    name = attrs.get("IncidentName")
    if name:
        parts.append(f"Wildfire incident: {name}.")

    category = attrs.get("IncidentTypeCategory")
    if category:
        parts.append(f"Type: {category}.")

    uid = attrs.get("UniqueFireIdentifier")
    if uid:
        parts.append(f"Unique Fire ID: {uid}.")

    daily_acres = attrs.get("DailyAcres")
    calc_acres = attrs.get("CalculatedAcres")
    if daily_acres is not None:
        parts.append(f"Daily acres burned: {daily_acres}.")
    if calc_acres is not None:
        parts.append(f"Calculated acres: {calc_acres}.")

    pct = attrs.get("PercentContained")
    if pct is not None:
        parts.append(f"Percent contained: {pct}%.")

    discovery = attrs.get("FireDiscoveryDateTime")
    if discovery:
        parts.append(f"Discovery date/time: {discovery}.")

    report_dt = attrs.get("ICS209ReportDateTime")
    if report_dt:
        parts.append(f"ICS-209 report date/time: {report_dt}.")

    state = attrs.get("POOState")
    county = attrs.get("POOCounty")
    if state:
        parts.append(f"State of origin: {state}.")
    if county:
        parts.append(f"County of origin: {county}.")

    cause = attrs.get("FireCause")
    if cause:
        parts.append(f"Fire cause: {cause}.")

    complexity = attrs.get("ComplexityLevel")
    if complexity:
        parts.append(f"Complexity level: {complexity}.")

    if geometry:
        x = geometry.get("x")
        y = geometry.get("y")
        if x is not None and y is not None:
            parts.append(f"Location (lon/lat): {x:.5f}, {y:.5f}.")

    return " ".join(parts) if parts else "Unknown wildfire incident."


def load_geojson_documents(path: Path) -> list[Document]:
    """Load GeoJSON features and convert each to a LangChain Document."""
    with open(path, "r") as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"Loaded {len(features)} features from {path.name}")

    documents = []
    for feature in features:
        text = feature_to_text(feature)
        attrs = feature.get("attributes", {})
        metadata = {
            "objectid": attrs.get("OBJECTID", 0),
            "incident_name": attrs.get("IncidentName", ""),
            "state": attrs.get("POOState", ""),
            "daily_acres": attrs.get("DailyAcres", 0.0),
            "percent_contained": attrs.get("PercentContained", 0.0),
            "fire_cause": attrs.get("FireCause", ""),
        }
        # ChromaDB rejects None metadata values — strip them
        metadata = {k: v for k, v in metadata.items() if v is not None}
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def build_vector_store(documents: list[Document]) -> Chroma:
    """Embed documents and persist them to ChromaDB."""
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding {len(documents)} documents and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="wildfires",
    )
    print(f"Vector store persisted to: {CHROMA_DIR}")
    return vectorstore


def main():
    documents = load_geojson_documents(GEOJSON_PATH)
    vectorstore = build_vector_store(documents)

    # Quick smoke-test: semantic search
    print("\n--- Smoke test: semantic search ---")
    queries = [
        "large wildfire in California",
        "fire with low containment percentage",
        "lightning caused fire",
    ]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            name = doc.metadata.get("incident_name", "Unknown")
            state = doc.metadata.get("state", "?")
            acres = doc.metadata.get("daily_acres", "?")
            print(f"  {i}. {name} ({state}) — {acres} acres")

    print("\nDone. Run query_geojson.py to perform interactive queries.")


if __name__ == "__main__":
    main()
