"""
Demo using the qdrant vector database to store and retrieve information about the rastertoolkit package.
"""
import os
import json
import qdrant_client
from qdrant_client import models
from sentence_transformers import SentenceTransformer

this_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
with open(os.path.join(this_dir, 'data', 'rastertoolkit_docs.jsonl'), 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# Create a qdrant client
client = qdrant_client.QdrantClient(":memory:")

# Create a sentence transformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Create a collection
client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

# Upload the docs
client.upload_points(
    collection_name="docs",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["docstring"]).tolist(), payload={k:doc[k] for k in ['name', 'type', 'file']}
        )
        for idx, doc in enumerate(docs)
    ],
)

# Ask a question
hits = client.query_points(
    collection_name="docs",
    query=encoder.encode("shape subdivide").tolist(),
    limit=3,
).points

# Print the results
for hit in hits:
    print(hit.payload, "score:", hit.score)