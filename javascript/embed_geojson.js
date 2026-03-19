/**
 * embed_geojson.js
 *
 * Demonstrates the core thesis: vector embedding GeoJSON features enables
 * semantic search for LLM agents — no SQL or custom query tools needed.
 *
 * Each wildfire incident is converted to a natural-language text document,
 * embedded with a local Xenova/transformers model, and stored in ChromaDB.
 *
 * Usage:
 *   npm run embed
 */

import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { Document } from "@langchain/core/documents";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..");
const GEOJSON_PATH = resolve(REPO_ROOT, "sample_wildfire.json");
const COLLECTION_NAME = "wildfires_js";
const CHROMA_URL = "http://localhost:8000"; // default ChromaDB server

// ---------------------------------------------------------------------------
// Convert a GeoJSON feature to a natural-language text description
// ---------------------------------------------------------------------------
function featureToText(feature) {
  const attrs = feature.attributes || {};
  const geometry = feature.geometry || {};
  const parts = [];

  if (attrs.IncidentName) parts.push(`Wildfire incident: ${attrs.IncidentName}.`);
  if (attrs.IncidentTypeCategory) parts.push(`Type: ${attrs.IncidentTypeCategory}.`);
  if (attrs.UniqueFireIdentifier) parts.push(`Unique Fire ID: ${attrs.UniqueFireIdentifier}.`);
  if (attrs.DailyAcres != null) parts.push(`Daily acres burned: ${attrs.DailyAcres}.`);
  if (attrs.CalculatedAcres != null) parts.push(`Calculated acres: ${attrs.CalculatedAcres}.`);
  if (attrs.PercentContained != null) parts.push(`Percent contained: ${attrs.PercentContained}%.`);
  if (attrs.FireDiscoveryDateTime) parts.push(`Discovery date/time: ${attrs.FireDiscoveryDateTime}.`);
  if (attrs.ICS209ReportDateTime) parts.push(`ICS-209 report date/time: ${attrs.ICS209ReportDateTime}.`);
  if (attrs.POOState) parts.push(`State of origin: ${attrs.POOState}.`);
  if (attrs.POOCounty) parts.push(`County of origin: ${attrs.POOCounty}.`);
  if (attrs.FireCause) parts.push(`Fire cause: ${attrs.FireCause}.`);
  if (attrs.ComplexityLevel) parts.push(`Complexity level: ${attrs.ComplexityLevel}.`);
  if (geometry.x != null && geometry.y != null) {
    parts.push(`Location (lon/lat): ${geometry.x.toFixed(5)}, ${geometry.y.toFixed(5)}.`);
  }

  return parts.length > 0 ? parts.join(" ") : "Unknown wildfire incident.";
}

// ---------------------------------------------------------------------------
// Load GeoJSON features and convert to LangChain Documents
// ---------------------------------------------------------------------------
function loadGeoJSONDocuments(path) {
  const raw = readFileSync(path, "utf-8");
  const data = JSON.parse(raw);
  const features = data.features || [];
  console.log(`Loaded ${features.length} features from ${path}`);

  return features.map((feature) => {
    const attrs = feature.attributes || {};
    return new Document({
      pageContent: featureToText(feature),
      metadata: {
        objectid: attrs.OBJECTID,
        incident_name: attrs.IncidentName || "",
        state: attrs.POOState || "",
        daily_acres: attrs.DailyAcres ?? null,
        percent_contained: attrs.PercentContained ?? null,
        fire_cause: attrs.FireCause || "",
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Build vector store
// ---------------------------------------------------------------------------
async function buildVectorStore(documents) {
  console.log("Loading embedding model (Xenova/all-MiniLM-L6-v2)...");
  const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
  });

  console.log(`Embedding ${documents.length} documents and storing in ChromaDB...`);
  const vectorstore = await Chroma.fromDocuments(documents, embeddings, {
    collectionName: COLLECTION_NAME,
    url: CHROMA_URL,
  });

  console.log(`Vector store created: collection '${COLLECTION_NAME}' on ${CHROMA_URL}`);
  return vectorstore;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const documents = loadGeoJSONDocuments(GEOJSON_PATH);
  const vectorstore = await buildVectorStore(documents);

  // Quick smoke-test
  console.log("\n--- Smoke test: semantic search ---");
  const queries = [
    "large wildfire in California",
    "fire with low containment percentage",
    "lightning caused fire",
  ];

  for (const query of queries) {
    console.log(`\nQuery: '${query}'`);
    const results = await vectorstore.similaritySearch(query, 3);
    results.forEach((doc, i) => {
      const { incident_name, state, daily_acres } = doc.metadata;
      console.log(`  ${i + 1}. ${incident_name} (${state}) — ${daily_acres} acres`);
    });
  }

  console.log("\nDone. Run 'npm run query' to perform interactive queries.");
}

main().catch(console.error);
