"""Flush qdrant DB"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
GRPC_PORT = os.environ.get("QDRANT_GRPC_PORT", "6334")

client = QdrantClient(
    host=HOST,
    grpc_port=GRPC_PORT,
    prefer_grpc=True,
)

collections = client.get_collections()
print(collections)

client.recreate_collection(
    "stuff",
    VectorParams(size=100, distance=Distance.COSINE),
)

collections = client.get_collections()
print(collections)

for collection in collections.collections:
    client.delete_collection(collection.name)

collections = client.get_collections()
print(collections)
