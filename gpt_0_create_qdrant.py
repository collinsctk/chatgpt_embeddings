from qdrant_client import QdrantClient
from qdrant_client.http import models

# 在下文中提醒如果数据量大, 需要使用专用的向量搜索引擎, 如Pinecone, Weaviate, Qdrant
# https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
"""
This indexing stage can be executed offline and only runs once to precompute the indexes for the dataset 
so that each piece of content can be retrieved later. Since this is a small example, we will store and search the 
embeddings locally. If you have a larger dataset, consider using a vector search engine like Pinecone, 
Weaviate or Qdrant to power the search.
"""

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
