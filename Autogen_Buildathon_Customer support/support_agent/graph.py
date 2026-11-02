"""Neo4j knowledge graph: build at ingestion time, read-only traversal at query time.

Schema (every node carries `product` so traversals never cross products):
    (:Product {slug, name})-[:HAS_DOCUMENT]->(:Document {key, name, doc_type, product})
    (:Document)-[:HAS_SECTION]->(:Section {id, title, text, summary, source_file, product})
    (:Section)-[:NEXT]->(:Section)
    (:Section)-[:MENTIONS]->(:Entity {key, name, product})
    (:Entity)-[:RELATED_TO {relation}]->(:Entity)

Query-time safety: the LLM only produces a *plan* (entity names, relation keywords,
hops 1-2). Code turns the plan into a parameterised Cypher template, validates it
against a write-keyword blocklist, and runs it in a READ transaction.
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from neo4j import READ_ACCESS, Driver, GraphDatabase

from .config import ENTITY_EXTRACT_SYSTEM, GRAPH_PLAN_SYSTEM, Settings, tokenize
from .models import Chunk, GraphPlan, SectionEntities

log = logging.getLogger(__name__)

# Any of these tokens => the query could mutate data or call admin procedures.
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s+dbms|"
    r"CALL\s+db\.(?!labels|relationshipTypes|propertyKeys)|apoc\.(create|merge|refactor|periodic|load|cypher)|"
    r"GRANT|REVOKE|DENY|ALTER|RENAME|START\s+DATABASE|STOP\s+DATABASE|IN\s+TRANSACTIONS)\b",
    re.IGNORECASE,
)


class UnsafeCypherError(ValueError):
    """Raised when a query is not provably read-only."""


@contextmanager
def get_driver(settings: Settings) -> Iterator[Driver]:
    """Neo4j driver as a context manager (TLS verified when using neo4j+s:// URIs)."""
    pwd = settings.neo4j_password.get_secret_value()
    if not pwd:
        raise RuntimeError("NEO4J_PASSWORD is not set in .env")
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, pwd),
                                  connection_timeout=10)
    try:
        driver.verify_connectivity()
        yield driver
    finally:
        driver.close()


# --------------------------------------------------------------------------- #
# Build (ingestion only — the only place writes happen)
# --------------------------------------------------------------------------- #
def extract_section_entities(title: str, text: str, settings: Settings) -> SectionEntities:
    """LLM entity/relation extraction for a section, cached by content hash."""
    from .ingest import cached_json
    from .llm import get_chat_llm

    def _run() -> dict:
        llm = get_chat_llm().with_structured_output(SectionEntities)
        res: SectionEntities = llm.invoke([
            SystemMessage(content=ENTITY_EXTRACT_SYSTEM),
            HumanMessage(content=f"Section: {title}\n\n{text[:6000]}"),
        ])
        return res.model_dump()

    return SectionEntities(**cached_json("entities", f"{title}|{text}", _run, settings))


def build_product_graph(driver: Driver, product_name: str, slug: str, chunks: list[Chunk],
                        records: dict[str, dict], settings: Settings) -> None:
    """(Re)build one product's subgraph. Old nodes for the product are removed first."""
    with driver.session(database=settings.neo4j_database) as session:
        session.execute_write(lambda tx: tx.run(
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE").consume())
        session.execute_write(lambda tx: tx.run(
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE").consume())
        session.execute_write(lambda tx: tx.run(
            "MATCH (n {product:$p}) DETACH DELETE n", p=slug).consume())
        session.execute_write(lambda tx: tx.run(
            "MERGE (p:Product {slug:$slug}) SET p.name=$name", slug=slug, name=product_name).consume())

        ordered = list(records.values())
        for rec in ordered:
            ents = extract_section_entities(rec["title"], rec["text"], settings)
            session.execute_write(_write_section, slug, rec, ents)
        # NEXT links between consecutive sections of the same document
        for a, b in zip(ordered, ordered[1:]):
            if a["source_file"] == b["source_file"]:
                session.execute_write(lambda tx, a=a, b=b: tx.run(
                    "MATCH (x:Section {id:$a}),(y:Section {id:$b}) MERGE (x)-[:NEXT]->(y)",
                    a=a["record_id"], b=b["record_id"]).consume())
    log.info("Graph built for %s (%d sections)", product_name, len(records))


def _write_section(tx, slug: str, rec: dict, ents: SectionEntities) -> None:
    tx.run(
        """
        MATCH (p:Product {slug:$slug})
        MERGE (d:Document {key:$dkey}) SET d.name=$file, d.doc_type=$dt, d.product=$slug
        MERGE (p)-[:HAS_DOCUMENT]->(d)
        MERGE (s:Section {id:$sid})
          SET s.title=$title, s.text=$text, s.summary=$summary, s.source_file=$file, s.product=$slug
        MERGE (d)-[:HAS_SECTION]->(s)
        WITH s
        UNWIND $entities AS en
          MERGE (e:Entity {key: $slug + '|' + en}) SET e.name = en, e.product = $slug
          MERGE (s)-[:MENTIONS]->(e)
        """,
        slug=slug, dkey=f"{slug}|{rec['source_file']}", file=rec["source_file"], dt=rec["doc_type"],
        sid=rec["record_id"], title=rec["title"], text=rec["text"][:8000], summary=rec.get("summary", ""),
        entities=[e.strip().lower() for e in ents.entities if e.strip()][:40],
    ).consume()
    for r in ents.relations[:40]:
        tx.run(
            """
            MERGE (a:Entity {key:$slug+'|'+$s}) SET a.name=$s, a.product=$slug
            MERGE (b:Entity {key:$slug+'|'+$t}) SET b.name=$t, b.product=$slug
            MERGE (a)-[r:RELATED_TO {relation:$rel}]->(b)
            """,
            slug=slug, s=r.source.strip().lower(), t=r.target.strip().lower(), rel=r.relation.strip().lower(),
        ).consume()


# --------------------------------------------------------------------------- #
# Read-only traversal
# --------------------------------------------------------------------------- #
def validate_read_only_cypher(query: str) -> str:
    """Reject any query that could write, delete or call admin procedures."""
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "''", query)  # ignore string literals
    if _WRITE_PATTERN.search(stripped):
        raise UnsafeCypherError(f"Write/admin clause detected in Cypher: {query[:120]}")
    if ";" in stripped.strip().rstrip(";"):
        raise UnsafeCypherError("Multiple statements are not allowed")
    if not re.match(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|CALL\s+db\.(labels|relationshipTypes))",
                    stripped, re.I):
        raise UnsafeCypherError("Query must start with a read clause")
    return query


@traceable(name="plan_graph_traversal")
def plan_graph_traversal(question: str, entities_hint: list[str]) -> GraphPlan:
    """LLM decides which entities/relations to traverse (structured output)."""
    from .llm import get_chat_llm

    llm = get_chat_llm().with_structured_output(GraphPlan)
    plan: GraphPlan = llm.invoke([
        SystemMessage(content=GRAPH_PLAN_SYSTEM),
        HumanMessage(content=f"Question: {question}\nEntities already detected: {entities_hint}"),
    ])
    plan.entities = [e.lower().strip() for e in plan.entities if e.strip()][:8]
    return plan


def build_traversal_query(hops: int) -> str:
    """Parameterised traversal template. `hops` is range-checked (1-2) before formatting."""
    if hops not in (1, 2):
        raise UnsafeCypherError("hops must be 1 or 2")
    return validate_read_only_cypher(f"""
        MATCH (s:Section {{product:$product}})-[:MENTIONS]->(e:Entity {{product:$product}})
        WHERE any(n IN $names WHERE e.name CONTAINS n OR n CONTAINS e.name)
        OPTIONAL MATCH (e)-[r:RELATED_TO*1..{hops}]-(e2:Entity {{product:$product}})
        WITH s, e, [rel IN coalesce(r, []) | rel.relation] AS rels, e2
        WITH s, collect(DISTINCT e.name) AS matched,
             collect(DISTINCT e.name + ' -' + coalesce(head(rels), 'related') + '-> ' + e2.name) AS facts
        OPTIONAL MATCH (s)-[:NEXT]-(nb:Section {{product:$product}})
        RETURN s.id AS section_id, s.title AS title, s.text AS text, s.source_file AS source_file,
               matched, [f IN facts WHERE f IS NOT NULL][..8] AS facts,
               collect(DISTINCT nb.id)[..2] AS neighbours
        ORDER BY size(matched) DESC
        LIMIT $limit
    """)


def run_read_query(driver: Driver, query: str, params: dict, database: str) -> list[dict]:
    """Execute a validated query in a READ session/transaction (server rejects writes too)."""
    validate_read_only_cypher(query)
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return session.execute_read(lambda tx: [dict(r) for r in tx.run(query, **params)])


@traceable(name="graph_retrieve")
def graph_retrieve(driver: Driver, product: str, question: str, entities_hint: list[str],
                   settings: Settings, limit: int = 8, plan: GraphPlan | None = None) -> list[dict]:
    """Plan (LLM) -> template -> validate -> read-only run. Returns section rows + facts."""
    plan = plan or plan_graph_traversal(question, entities_hint)
    names = plan.entities or [t for t in tokenize(question) if len(t) > 3][:8]
    names += [k.lower() for k in plan.relation_keywords if k.strip()][:4]
    if not names:
        return []
    query = build_traversal_query(plan.hops)
    return run_read_query(driver, query, {"product": product, "names": names, "limit": limit},
                          settings.neo4j_database)


def graph_sample(driver: Driver, product: str, settings: Settings, limit: int = 50) -> list[dict]:
    """Admin view: entity relations for a product (read-only)."""
    q = validate_read_only_cypher("""
        MATCH (s:Section {product:$product})-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (e)-[r:RELATED_TO]->(e2:Entity)
        RETURN s.source_file AS source_file, s.title AS section, e.name AS entity,
               r.relation AS relation, e2.name AS related_entity
        LIMIT $limit
    """)
    return run_read_query(driver, q, {"product": product, "limit": limit}, settings.neo4j_database)
