"""Knowledge graph: build (write path, ingestion only) and read-only traversal.

Read path safety is enforced twice:
  1. `validate_read_only_cypher` rejects any write/admin clause in code.
  2. Queries run in a READ_ACCESS session via `execute_read`, so Neo4j itself
     refuses writes even if validation were bypassed.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable
from pydantic import BaseModel, Field

from hybrid_rag import config


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(settings: config.Settings):
    """Neo4j driver from .env credentials (TLS verified via the URI scheme,
    e.g. neo4j+s:// for Aura)."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


# ---------------------------------------------------------------------------
# Build (write) - only called from ingestion
# ---------------------------------------------------------------------------

class GEntity(BaseModel):
    name: str
    type: str = "Thing"


class GRelation(BaseModel):
    source: str
    relation: str = Field(description="UPPER_SNAKE_CASE")
    target: str


class GraphExtraction(BaseModel):
    entities: list[GEntity] = Field(default_factory=list)
    relations: list[GRelation] = Field(default_factory=list)


@traceable(name="extract_graph_from_chunk")
def extract_graph_from_chunk(doc: Document, llm) -> GraphExtraction:
    """LLM -> entities and relations for one chunk (structured output)."""
    extractor = llm.with_structured_output(GraphExtraction)
    return extractor.invoke(f"{config.GRAPH_EXTRACT_PROMPT}\n\nTEXT:\n{doc.page_content[:4000]}")


def normalise_relation(rel: str) -> str:
    """Relation types are stored as a property; keep them tidy and safe."""
    return re.sub(r"[^A-Z0-9_]", "_", rel.upper()).strip("_")[:60] or "RELATED_TO"


_WRITE_CHUNK = """
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.text = $text, c.source = $source, c.doc_id = $doc_id
WITH c
UNWIND $entities AS e
  MERGE (n:Entity {name: e.name}) ON CREATE SET n.type = e.type
  MERGE (c)-[:MENTIONS]->(n)
"""
_WRITE_REL = """
UNWIND $relations AS r
  MERGE (a:Entity {name: r.source})
  MERGE (b:Entity {name: r.target})
  MERGE (a)-[x:REL {type: r.relation}]->(b)
  SET x.chunk_id = $chunk_id
"""


@traceable(name="build_graph")
def build_graph(driver, docs: list[Document], llm, database: str = "neo4j",
                max_chunks: int = 80) -> dict:
    """Extract triples from chunks and MERGE them into Neo4j (idempotent)."""
    with driver.session(database=database) as s:
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    n_rel = 0
    for doc in docs[:max_chunks]:
        try:
            g = extract_graph_from_chunk(doc, llm)
        except Exception:
            continue
        ents = [{"name": e.name.strip(), "type": e.type} for e in g.entities if e.name.strip()]
        rels = [{"source": r.source.strip(), "target": r.target.strip(),
                 "relation": normalise_relation(r.relation)} for r in g.relations
                if r.source.strip() and r.target.strip()]
        meta = doc.metadata
        params = dict(chunk_id=meta["chunk_id"], text=doc.page_content,
                      source=meta.get("source", ""), doc_id=meta.get("doc_id", ""))
        with driver.session(database=database) as s:
            s.execute_write(lambda tx: tx.run(_WRITE_CHUNK, entities=ents, **params).consume())
            if rels:
                s.execute_write(lambda tx: tx.run(_WRITE_REL, relations=rels,
                                                  chunk_id=meta["chunk_id"]).consume())
        n_rel += len(rels)
    return {"chunks": min(len(docs), max_chunks), "relations": n_rel}


# ---------------------------------------------------------------------------
# Read-only traversal
# ---------------------------------------------------------------------------

class UnsafeCypherError(ValueError):
    pass


_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL|GRANT|REVOKE|DENY|"
    r"ALTER|RENAME|START|STOP|TERMINATE|IMPORT|USE)\b|apoc\.|dbms\.|db\.",
    re.IGNORECASE)
_ALLOWED_START = re.compile(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND)\b", re.IGNORECASE)
_STRING_LIT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.IGNORECASE)


def validate_read_only_cypher(query: str, max_rows: int = 25) -> str:
    """Return a safe version of `query` or raise UnsafeCypherError.
    String literals are blanked before the keyword scan so a name such as
    'Create Jobs Scheme' is not mistaken for a CREATE clause. A LIMIT is
    enforced (added, or reduced to `max_rows`)."""
    q = re.sub(r"//.*?$|/\*.*?\*/", " ", query or "", flags=re.S | re.M).strip()
    q = re.sub(r"^```(?:cypher)?|```$", "", q, flags=re.I).strip().rstrip(";").strip()
    if not q:
        raise UnsafeCypherError("Empty query")
    scan = _STRING_LIT.sub("''", q)
    if ";" in scan:
        raise UnsafeCypherError("Multiple statements are not allowed")
    if not _ALLOWED_START.match(scan):
        raise UnsafeCypherError("Query must start with MATCH / OPTIONAL MATCH / WITH / UNWIND")
    bad = _FORBIDDEN.search(scan)
    if bad:
        raise UnsafeCypherError(f"Forbidden clause: {bad.group(0)}")
    if not re.search(r"\bRETURN\b", scan, re.I):
        raise UnsafeCypherError("Query must RETURN something")
    m = _LIMIT.search(q)
    if m:
        if int(m.group(1)) > max_rows:
            q = q[:m.start()] + f"LIMIT {max_rows}"
    else:
        q = f"{q}\nLIMIT {max_rows}"
    return q


def run_read_query(driver, query: str, params: dict | None = None,
                   database: str = "neo4j", max_rows: int = 25) -> list[dict]:
    """Validate, then execute in a READ session (database-enforced read-only)."""
    from neo4j import READ_ACCESS

    safe = validate_read_only_cypher(query, max_rows)
    with driver.session(database=database, default_access_mode=READ_ACCESS) as s:
        return s.execute_read(lambda tx: tx.run(safe, params or {}).data())


class CypherPlan(BaseModel):
    cypher: str = Field(description="A single read-only Cypher query")


@traceable(name="plan_cypher", run_type="llm")
def plan_cypher(question: str, entities: list[str], llm, limit: int = 25) -> str:
    """LLM plans the graph traversal (Cypher) - validated before execution."""
    planner = llm.with_structured_output(CypherPlan)
    prompt = config.CYPHER_PROMPT.format(question=question, entities=entities or "[]", limit=limit)
    return planner.invoke(prompt).cypher


ENTITY_TRAVERSAL = """
UNWIND $names AS q
MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower(q)
MATCH p = (e)-[:REL*1..2]-(o:Entity)
UNWIND relationships(p) AS r
WITH DISTINCT startNode(r) AS a, r, endNode(r) AS b
RETURN a.name AS source, r.type AS relation, b.name AS target, r.chunk_id AS chunk_id
"""


def traverse_entities(driver, names: list[str], database: str = "neo4j", max_rows: int = 25) -> list[dict]:
    """Deterministic fallback: 1-2 hop neighbourhood of matched entity names."""
    names = [n for n in names if n and len(n) > 2]
    if not names:
        return []
    return run_read_query(driver, ENTITY_TRAVERSAL, {"names": names}, database, max_rows)


@traceable(name="graph_search", run_type="retriever")
def graph_search(driver, question: str, entities: list[str], llm,
                 settings: config.Settings) -> dict[str, Any]:
    """LLM-planned Cypher first; if it is unsafe, fails or returns nothing,
    fall back to entity-neighbourhood traversal. Returns rows + query used."""
    out: dict[str, Any] = {"cypher": "", "rows": [], "error": ""}
    if llm is not None:
        try:
            cypher = plan_cypher(question, entities, llm, settings.graph_max_rows)
            out["cypher"] = validate_read_only_cypher(cypher, settings.graph_max_rows)
            out["rows"] = run_read_query(driver, out["cypher"], None, settings.neo4j_database,
                                         settings.graph_max_rows)
        except Exception as exc:  # unsafe or failing query -> fallback
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    if not out["rows"]:
        names = entities or [w for w in config.tokenize(question) if len(w) > 3][:5]
        out["cypher"] = ENTITY_TRAVERSAL.strip()
        try:
            out["rows"] = traverse_entities(driver, names, settings.neo4j_database, settings.graph_max_rows)
        except Exception as exc:
            out["error"] += f" | fallback: {exc}"[:300]
    return out


def graph_rows_to_text(rows: list[dict]) -> str:
    """Rows -> 'A -[REL]-> B' lines for the LLM context."""
    lines = []
    for r in rows:
        if {"source", "relation", "target"} <= r.keys():
            lines.append(f"{r['source']} -[{r['relation']}]-> {r['target']}")
        else:
            lines.append(", ".join(f"{k}={v}" for k, v in r.items()))
    return "\n".join(dict.fromkeys(lines))


def graph_stats(driver, database: str = "neo4j") -> dict:
    rows = run_read_query(driver, "MATCH (e:Entity) WITH count(e) AS entities "
                          "OPTIONAL MATCH ()-[r:REL]->() RETURN entities, count(r) AS relations",
                          database=database, max_rows=1)
    return rows[0] if rows else {"entities": 0, "relations": 0}
