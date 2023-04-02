from qdrant_client import QdrantClient
from qdrant_client.http import models

collection_name = "qyt_gpt_collection"

qdrant_url = "http://localhost:6333"

client = QdrantClient(qdrant_url)

dimension = 1536

if __name__ == "__main__":
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )

    # 使用游览器访问 http://localhost:6333/collections
    # {"result":{"collections":[{"name":"example_collection"},{"name":"qyt_gpt_collection"},
    # {"name":"qytang_gpt"}]},"status":"ok","time":0.000035}
