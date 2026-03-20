/**
 * query_geojson.js
 *
 * Interactive semantic search over the vector-embedded wildfire GeoJSON data.
 * Run embed_geojson.js first to populate the ChromaDB collection.
 *
 * Usage:
 *   npm run query                         # interactive mode
 *   node query_geojson.js "your query"    # single query
 */

import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { createInterface } from "readline";

const COLLECTION_NAME = "wildfires_js";
const CHROMA_URL = "http://localhost:8000";
const DEFAULT_K = 5;

// ---------------------------------------------------------------------------
// Load vector store
// ---------------------------------------------------------------------------
async function loadVectorStore() {
  console.log("Loading embedding model...");
  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });

  const vectorstore = new Chroma(embeddings, {
    collectionName: COLLECTION_NAME,
    url: CHROMA_URL,
  });

  return vectorstore;
}

// ---------------------------------------------------------------------------
// Search and display results
// ---------------------------------------------------------------------------
async function search(vectorstore, query, k = DEFAULT_K) {
  console.log(`\nQuery: '${query}'`);
  console.log("-".repeat(60));

  const results = await vectorstore.similaritySearchWithScore(query, k);
  results.forEach(([doc, score], i) => {
    const { incident_name, state, daily_acres, percent_contained, fire_cause } = doc.metadata;
    console.log(
      `  ${i + 1}. [${score.toFixed(4)}] ${incident_name} (${state})\n` +
        `       Acres: ${daily_acres} | Contained: ${percent_contained}% | Cause: ${fire_cause}`
    );
    console.log(`       ${doc.pageContent.slice(0, 120)}...`);
  });
}

// ---------------------------------------------------------------------------
// Interactive loop
// ---------------------------------------------------------------------------
async function interactiveLoop(vectorstore) {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("\nInteractive wildfire search (type 'quit' to exit)");
  console.log("=".repeat(60));

  const ask = () => {
    rl.question("\nEnter search query: ", async (query) => {
      query = query.trim();
      if (!query || query.toLowerCase() === "quit" || query.toLowerCase() === "exit") {
        rl.close();
        return;
      }
      await search(vectorstore, query);
      ask();
    });
  };

  ask();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const vectorstore = await loadVectorStore();

  const cliQuery = process.argv[2];
  if (cliQuery) {
    await search(vectorstore, cliQuery);
  } else {
    await interactiveLoop(vectorstore);
  }
}

main().catch(console.error);
