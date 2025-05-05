import chromadb
import pandas as pd

# Connect to your ChromaDB host
client = chromadb.HttpClient(host="170.64.231.135", port=8000)

# Load your collection
collection = client.get_collection(name="ghg_collection")

# Fetch all documents (adjust limit if needed)
results = collection.get(include=["documents", "metadatas", "embeddings"], limit=10)

# Convert to DataFrame
df = pd.DataFrame(
    {
        "document": results["documents"],
        "metadata": results["metadatas"],
        "embedding": results["embeddings"],
    }
)

# Save to CSV
df.to_csv("ghg_chunks_export.csv", index=False)

print("✅ Exported to ghg_chunks_export.csv")
