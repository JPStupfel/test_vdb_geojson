"""
query_geojson.py

Interactive semantic search over the vector-embedded wildfire GeoJSON data.
Run embed_geojson.py first to build the ChromaDB vector store.

Usage:
    python query_geojson.py [--query "your search query"] [--k 5]
"""

import argparse
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


def load_vector_store() -> Chroma:
    """Load the persisted ChromaDB vector store."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}.\n"
            "Run embed_geojson.py first to build the index."
        )

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="wildfires",
    )
    return vectorstore


def search(vectorstore: Chroma, query: str, k: int = 5) -> None:
    """Perform a similarity search and print results."""
    print(f"\nQuery: '{query}'")
    print("-" * 60)

    results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        name = meta.get("incident_name", "Unknown")
        state = meta.get("state", "?")
        acres = meta.get("daily_acres", "?")
        contained = meta.get("percent_contained", "?")
        cause = meta.get("fire_cause", "?")
        print(
            f"  {i}. [{score:.4f}] {name} ({state})\n"
            f"       Acres: {acres} | Contained: {contained}% | Cause: {cause}"
        )
        print(f"       {doc.page_content[:120]}...")


def interactive_loop(vectorstore: Chroma, k: int) -> None:
    """Run an interactive query loop."""
    print("\nInteractive wildfire search (type 'quit' to exit)")
    print("=" * 60)
    while True:
        query = input("\nEnter search query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            search(vectorstore, query, k=k)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over vector-embedded wildfire GeoJSON data"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Search query (omit for interactive mode)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    args = parser.parse_args()

    vectorstore = load_vector_store()

    if args.query:
        search(vectorstore, args.query, k=args.k)
    else:
        interactive_loop(vectorstore, k=args.k)


if __name__ == "__main__":
    main()
