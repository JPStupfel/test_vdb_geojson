"""
chat_agent.py

Interactive RAG chat agent over vector-embedded wildfire GeoJSON data.
Retrieves relevant wildfire features via semantic search and uses an LLM
to synthesize natural-language answers.

Prerequisites:
    1. Run embed_geojson.py first to build the ChromaDB vector store.
    2. Set OPENAI_API_KEY in your environment.

Usage:
    python chat_agent.py                    # interactive chat
    python chat_agent.py --query "..."      # single question
    python chat_agent.py --retrieval-only   # skip LLM, show raw retrieval
"""

import argparse
import os
import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

SYSTEM_PROMPT = """\
You are a wildfire data analyst. You answer questions about wildfire incidents
using ONLY the retrieved context below. Each context chunk describes one
wildfire incident with attributes like name, location, acres burned,
containment percentage, cause, and coordinates.

Rules:
- Base your answer strictly on the provided context.
- If the context doesn't contain enough information, say so.
- When listing fires, include name, state, acres, and containment if available.
- Be concise but thorough.

Context:
{context}
"""


def load_vector_store():
    if not CHROMA_DIR.exists():
        print(f"Error: Vector store not found at {CHROMA_DIR}")
        print("Run embed_geojson.py first to build the index.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="wildfires",
    )


def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    )


def retrieval_only(vectorstore, query, k=8):
    """Show raw retrieval results without LLM."""
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        print(
            f"  {i}. [{score:.4f}] {m.get('incident_name', '?')} "
            f"({m.get('state', '?')}) — {m.get('daily_acres', '?')} acres"
        )
    print()


def build_rag_chain(vectorstore, k=8):
    """Build a LangChain RAG chain: retriever → prompt → LLM → output."""
    from langchain_openai import ChatOpenAI

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def chat_loop(chain, vectorstore, show_sources):
    """Interactive chat loop."""
    print("\n🔥 Wildfire Data Chat Agent")
    print("=" * 60)
    print("Ask questions about wildfire incidents. Type 'quit' to exit.")
    if show_sources:
        print("(Sources will be shown below each answer)")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = chain.invoke(question)
        print(f"\nAgent: {answer}\n")

        if show_sources:
            docs = vectorstore.similarity_search(question, k=5)
            print("  Sources:")
            for i, doc in enumerate(docs, 1):
                m = doc.metadata
                print(
                    f"    {i}. {m.get('incident_name', '?')} "
                    f"({m.get('state', '?')}) — {m.get('daily_acres', '?')} acres"
                )
            print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG chat agent over vector-embedded wildfire GeoJSON"
    )
    parser.add_argument("--query", "-q", type=str, help="Single question (omit for interactive)")
    parser.add_argument("--k", type=int, default=8, help="Number of docs to retrieve (default: 8)")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM, show raw retrieval")
    parser.add_argument("--sources", action="store_true", help="Show source documents with answers")
    args = parser.parse_args()

    print("Loading vector store...")
    vectorstore = load_vector_store()

    if args.retrieval_only:
        if args.query:
            retrieval_only(vectorstore, args.query, k=args.k)
        else:
            print("\nRetrieval-only mode (type 'quit' to exit)")
            print("=" * 60)
            while True:
                try:
                    q = input("\nQuery: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in ("quit", "exit", "q", ""):
                    break
                retrieval_only(vectorstore, q, k=args.k)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set.")
        print("Options:")
        print("  1. export OPENAI_API_KEY='sk-...' and rerun")
        print("  2. Use --retrieval-only mode (no LLM needed)")
        sys.exit(1)

    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore, k=args.k)

    if args.query:
        answer = chain.invoke(args.query)
        print(f"\n{answer}")
        if args.sources:
            docs = vectorstore.similarity_search(args.query, k=5)
            print("\nSources:")
            for i, doc in enumerate(docs, 1):
                m = doc.metadata
                print(f"  {i}. {m.get('incident_name', '?')} ({m.get('state', '?')})")
    else:
        chat_loop(chain, vectorstore, show_sources=args.sources)


if __name__ == "__main__":
    main()
