import chromadb

# Connect to persistent ChromaDB store
client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection("drug_chunks")

# Add a Humira dosage chunk
collection.add(
    ids=["humira~35~2~a1b2c3d4"],
    documents=["Humira recommended dosage is 40mg every other week, subcutaneous injection."],
    metadatas=[{
        "drug": "Humira",
        "source_file": "humira.pdf",
        "page": 35,
        "section": "Dosing",
        "chunk_start": 1200,
        "chunk_id": "humira~35~2~a1b2c3d4"
    }]
)

print("✅ Sample chunk inserted")
