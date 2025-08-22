# -----------------------------
# Retriever Agent - ChromaDB Search
# -----------------------------
import chromadb

def search_db(entities: list[str], intents: list[str], top_k: int = 5):
    # Connect to ChromaDB (persistent or in-memory)
    client = chromadb.PersistentClient(path="chroma_store")

    # Get / create collection
    collection = client.get_or_create_collection("drug_chunks")

    query_texts = entities + intents

    # Run semantic search
    results = collection.query(
        query_texts=query_texts,
        n_results=top_k,
    )

    retrievals = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        retrievals.append({
            "text": results["documents"][0][i],
            "metadata": metadata,
            "citation": f"{metadata.get('source_file')} (Section: {metadata.get('section')}, Page: {metadata.get('page')})",
            "raw_distance": results["distances"][0][i],
            "score": 1 / (1 + results["distances"][0][i])  # normalized score
        })

    return {"retrievals": retrievals}
