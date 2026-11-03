# Database / persistent storage

This folder holds state that must survive container recreation.

- `storage/support_memory.db` - SQLite ticket/conversation memory
- `storage/ticket_log.csv` - local ticket log mirror
- `storage/indexes/` - FAISS/BM25 retrieval indexes
- `storage/cache/` - generated retrieval/OCR/entity caches
- `neo4j/` - Neo4j graph database files created at runtime

The backend container mounts `database/storage` at `/app/storage`.
The Neo4j container mounts `database/neo4j` at `/data`.
