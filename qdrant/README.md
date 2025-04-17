# README

Install requirements

```bash
uv pip install -r requirements.txt
``

Run server:
```bash
QDRANT_LOCAL_PATH="/home/krosenfeld/projects/RasterToolkit/qdrant/data/rastertoolkit_docs.db" \
COLLECTION_NAME="my-docs" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
uvx mcp-server-qdrant
```