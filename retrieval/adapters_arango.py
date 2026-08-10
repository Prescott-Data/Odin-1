from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict, Any, NamedTuple

from .adapters import GraphAccessor, NodeId, RelId


class EdgeView(NamedTuple):
    neighbor_id: NodeId
    relation: RelId
    weight: float                  # structural effective weight
    edge_id: str
    valid_from: Optional[str]
    valid_to: Optional[str]
    status: Optional[str]
    raw_confidence: Optional[float]
    npll_posterior: Optional[float]
    calibration: Optional[float]
    sources: List[str]             # doc/text ids from inline fields & EXTRACTED_FROM


class ArangoCommunityAccessor(GraphAccessor):
    """
    Arango-backed GraphAccessor for a single community.

    Defaults match your schema:
      - nodes:  ExtractedEntities
      - edges:  ExtractedRelationships  (field: relationship, created_at)
      - community via mapping: EntityCommunities(entity_id, community_id)
      - provenance: inline fields + EXTRACTED_FROM (entity -> Documents/TextBlocks)

    Structural weight only by default:
        w_struct = base_weight * type_prior(relation) * recency_decay
    (Set fuse_edge_confidence=True to multiply raw_confidence * npll_posterior * calibration in-adapter.)
    """

    def __init__(
        self,
        db,
        community_id: str,
        # Collections
        nodes_collection: str = "ExtractedEntities",
        edges_collection: str = "ExtractedRelationships",
        # Core field names
        relation_property: str = "relationship",
        weight_property: str = "weight",
        node_type_property: str = "type",
        # Time fields
        edge_timestamp_property: str = "created_at",
        edge_valid_from_property: Optional[str] = "valid_from",
        edge_valid_to_property: Optional[str] = "valid_to",
        edge_status_property: Optional[str] = "status",
        # Community scoping (mapping mode by default)
        community_mode: str = "mapping",  # "mapping" | "property"
        community_property: str = "community_id",  # only used if community_mode == "property"
        membership_collection: str = "EntityCommunities",
        membership_entity_field: str = "entity_id",
        membership_community_field: str = "community_id",
        # Dynamic constraints
        allowed_relations: Optional[List[str]] = None,
        disallowed_relations: Optional[List[str]] = None,
        allowed_neighbor_types: Optional[List[str]] = None,
        # Time filters
        time_window: Optional[Tuple[str, str]] = None,   # (start_iso, end_iso)
        as_of: Optional[str] = None,                     # ISO timestamp for "as of"
        current_only: bool = False,                      # respect valid_from/valid_to around as_of
        recency_half_life_days: Optional[float] = 90.0,  # None disables recency decay
        # Priors
        type_priors: Optional[Dict[str, float]] = None,  # e.g., {"assessor": 1.1}
        # Provenance
        edge_provenance_fields: Optional[List[str]] = None,  # defaults: ["source_document_id","source_text_id"]
        provenance_edge_collection: Optional[str] = "EXTRACTED_FROM",
        provenance_target_collections: Optional[List[str]] = None,  # defaults: ["Documents","TextBlocks"]
        # Confidence fusion (usually False; you do NPLL in engine)
        fuse_edge_confidence: bool = False,
        missing_confidence_prior: float = 1.0,
        edge_raw_confidence_property: Optional[str] = "raw_confidence",
        edge_npll_posterior_property: Optional[str] = "npll_posterior",
        edge_calibration_property: Optional[str] = "calibration",
        # Performance
        aql_batch_size: int = 1000,
        aql_stream: bool = True,
        outbound_index_hint: Optional[str] = None,  # e.g. "edges_from_rel_ts"
        inbound_index_hint: Optional[str] = None,   # e.g. "edges_to_rel_ts"
        # Bridge / GNN integration
        bridge_collection: str = "BridgeEntities",
        affinity_collection: str = "CommunityAffinity",
        algorithm: str = "gnn",  # Default to GNN as per pipeline
    ):
        self.db = db
        self._cid = community_id
        self.bridge_col = bridge_collection
        self.affinity_col = affinity_collection
        self.algorithm = algorithm
        self._bridge_cache: Dict[str, Optional[dict]] = {}
        self._affinity_cache: Dict[str, float] = {}

        self.nodes_col = nodes_collection
        self.edges_col = edges_collection

        self.rel_prop = relation_property
        self.w_prop = weight_property
        self.node_type_prop = node_type_property

        self.ts_prop = edge_timestamp_property
        self.edge_valid_from_prop = edge_valid_from_property
        self.edge_valid_to_prop = edge_valid_to_property
        self.edge_status_prop = edge_status_property

        self.community_mode = community_mode
        self.community_prop = community_property
        self.membership_col = membership_collection
        self.memb_ent_field = membership_entity_field
        self.memb_com_field = membership_community_field

        self.allowed_relations = allowed_relations
        self.disallowed_relations = disallowed_relations
        self.allowed_neighbor_types = allowed_neighbor_types

        self.time_window = time_window
        self.as_of = as_of
        self.current_only = current_only
        self.recency_half_life_days = recency_half_life_days

        self.type_priors = type_priors or {}

        self.edge_prov_fields = edge_provenance_fields or ["source_document_id", "source_text_id"]
        self.prov_edges_col = provenance_edge_collection
        self.prov_target_cols = provenance_target_collections or ["Documents", "TextBlocks"]

        self.fuse_edge_confidence = fuse_edge_confidence
        self.missing_confidence_prior = missing_confidence_prior
        self.edge_raw_conf_prop = edge_raw_confidence_property
        self.edge_npll_post_prop = edge_npll_posterior_property
        self.edge_calibration_prop = edge_calibration_property

        self.aql_batch_size = aql_batch_size
        self.aql_stream = aql_stream
        self.outbound_index_hint = outbound_index_hint
        self.inbound_index_hint = inbound_index_hint

    # --------------------------
    # Back-compatible core API
    # --------------------------
    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        for ev in self._iter_neighbors(node, direction="OUTBOUND", rich=True):
            yield ev.neighbor_id, ev.relation, ev.weight

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        for ev in self._iter_neighbors(node, direction="INBOUND", rich=True):
            yield ev.neighbor_id, ev.relation, ev.weight

    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        """
        Return all node IDs in this community.
        - mapping mode: EntityCommunities -> entity_id
        - property mode: filter ExtractedEntities by community_id field (if you add it)
        - none mode: return all nodes
        """
        cid = community_id or self._cid
        if self.community_mode == "property":
            aql = f"""
            FOR v IN {self.nodes_col}
              FILTER v.{self.community_prop} == @cid
              RETURN v._id
            """
            cursor = self.db.aql.execute(
                aql, bind_vars={"cid": cid}, batch_size=self.aql_batch_size, stream=self.aql_stream
            )
        elif self.community_mode == "mapping":
            aql = f"""
            FOR m IN @@mcol
              FILTER m[@m_com] == @cid
              RETURN m[@m_ent]
            """
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "cid": cid,
                    "@mcol": self.membership_col,
                    "m_ent": self.memb_ent_field,
                    "m_com": self.memb_com_field,
                },
                batch_size=self.aql_batch_size,
                stream=self.aql_stream,
            )
        else:  # community_mode == "none"
            aql = f"""
            FOR v IN {self.nodes_col}
              RETURN v._id
            """
            cursor = self.db.aql.execute(
                aql, batch_size=self.aql_batch_size, stream=self.aql_stream
            )
        for vid in cursor:
            yield vid

    def degree(self, node: NodeId) -> int:
        """Out-degree (fast)."""
        hint_clause = (
            "OPTIONS { indexHint: @idx, forceIndexHint: true }" if self.outbound_index_hint else ""
        )
        aql = f"""
        RETURN LENGTH(
          FOR e IN {self.edges_col}
            {hint_clause}
            FILTER e._from == @node
            RETURN 1
        )
        """
        bind = {"node": node}
        if self.outbound_index_hint:
            bind["idx"] = self.outbound_index_hint
        cur = self.db.aql.execute(aql, bind_vars=bind)
        return int(list(cur)[0] or 0)

    # --------------------------
    # Rich neighbor variants
    # --------------------------
    def iter_out_rich(self, node: NodeId) -> Iterable[EdgeView]:
        yield from self._iter_neighbors(node, direction="OUTBOUND", rich=True)

    def iter_in_rich(self, node: NodeId) -> Iterable[EdgeView]:
        yield from self._iter_neighbors(node, direction="INBOUND", rich=True)

    # --------------------------
    # Provenance helpers
    # --------------------------
    def get_edge_provenance(self, edge_id: str) -> List[str]:
        """
        Return provenance targets for a relationship edge:
          - inline fields (source_document_id, source_text_id)
          - EXTRACTED_FROM edges for either endpoint entity
        """
        # Build the provenance edges clause safely (avoid nested f-strings)
        prov_edges_clause = (
            f"""
            FOR p IN {self.prov_edges_col}
              FILTER p._from IN [e._from, e._to]
              RETURN p._to
            """
            if self.prov_edges_col else "[]"
        )

        aql = f"""
        LET e = DOCUMENT(@eid)
        LET inline_candidates = [{", ".join([f"e['{f}']" for f in self.edge_prov_fields])}]
        LET inline = (
          FOR x IN inline_candidates
            FILTER x != null
            RETURN x
        )
        LET via_edges = (
          {prov_edges_clause}
        )
        RETURN UNIQUE(APPEND(inline, via_edges))
        """
        cur = self.db.aql.execute(aql, bind_vars={"eid": edge_id}, batch_size=self.aql_batch_size, stream=self.aql_stream)
        out = list(cur)
        return out[0] if out else []

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if fields:
            proj = ", ".join([f"{f}: d.{f}" for f in fields])
            aql = f"LET d = DOCUMENT(@id) RETURN {{ _id: d._id, {proj} }}"
        else:
            aql = "RETURN DOCUMENT(@id)"
        cur = self.db.aql.execute(aql, bind_vars={"id": node_id})
        res = list(cur)
        return res[0] if res else {}

    # --------------------------
    # Stats / quick analytics
    # --------------------------
    @staticmethod
    def get_top_n_entities_by_degree(
        db,
        edges_collection: str = "ExtractedRelationships",
        limit: Optional[int] = None,
        time_window: Optional[Tuple[str, str]] = None,
        time_property: str = "created_at",
    ) -> List[dict]:
        bind: Dict[str, Any] = {}
        where = ""
        if time_window:
            where = "FILTER HAS(e, @ts) AND e[@ts] >= @start_ts AND e[@ts] <= @end_ts"
            bind.update({"ts": time_property, "start_ts": time_window[0], "end_ts": time_window[1]})
        limit_clause = "LIMIT @lim" if limit else ""
        if limit:
            bind["lim"] = limit
        aql = f"""
        FOR e IN {edges_collection}
          {where}
          COLLECT entity = e._from WITH COUNT INTO degree
          SORT degree DESC
          {limit_clause}
          RETURN {{ "entity": entity, "degree": degree }}
        """
        return list(db.aql.execute(aql, bind_vars=bind))

    @staticmethod
    def get_entity_type_counts(
        db,
        nodes_collection: str = "ExtractedEntities",
        type_property: str = "type"
    ) -> List[dict]:
        aql = f"""
        FOR doc IN {nodes_collection}
          COLLECT t = doc.{type_property} WITH COUNT INTO c
          SORT c DESC
          RETURN {{ "type": t, "count": c }}
        """
        return list(db.aql.execute(aql))

    @staticmethod
    def get_relationship_type_counts(
        db,
        edges_collection: str = "ExtractedRelationships",
        relation_property: str = "relationship",
        time_window: Optional[Tuple[str, str]] = None,
        time_property: str = "created_at",
    ) -> List[dict]:
        bind: Dict[str, Any] = {"rel_prop": relation_property}
        where = "FILTER HAS(rel, @rel_prop)"
        if time_window:
            where += " AND HAS(rel, @ts) AND rel[@ts] >= @start_ts AND rel[@ts] <= @end_ts"
            bind.update({"ts": time_property, "start_ts": time_window[0], "end_ts": time_window[1]})
        aql = f"""
        FOR rel IN {edges_collection}
          {where}
          COLLECT t = rel[@rel_prop] WITH COUNT INTO c
          SORT c DESC
          RETURN {{ "type": t, "count": c }}
        """
        return list(db.aql.execute(aql, bind_vars=bind))

    @staticmethod
    def get_community_summaries(
        db,
        communities_collection: str = "Communities",
        limit: Optional[int] = None,
        skip: int = 0,
        require_summary: bool = True
    ) -> List[dict]:
        filter_clause = "FILTER c.summary != null AND c.summary != ''" if require_summary else "FILTER c.summary == null OR c.summary == ''"
        limit_clause = "LIMIT @skip, @limit" if limit is not None else ""
        bind: Dict[str, Any] = {}
        if limit is not None:
            bind.update({"skip": skip, "limit": limit})
            aql = f"""
            FOR c IN {communities_collection}
              {filter_clause}
              SORT c.community_id ASC
          {limit_clause}
              RETURN {{ id: c.community_id, summary: c.summary, size: c.size, level: c.level }}
            """
        return list(db.aql.execute(aql, bind_vars=bind))

    @staticmethod
    def get_unique_table_headers(
        db,
        tables_collection: str = "Tables",
        headers_property: str = "headers"
    ) -> List[List[str]]:
        aql = f"""
        FOR t IN {tables_collection}
          FILTER HAS(t, @hp)
          COLLECT h = t[@hp]
          RETURN h
        """
        return list(db.aql.execute(aql, bind_vars={"hp": headers_property}))

    # --------------------------
    # Bridge / GNN Integration Methods (Mirrored from GlobalGraphAccessor)
    # --------------------------

    def is_bridge(self, entity_key: str) -> Optional[dict]:
        """
        Check if an entity is a bridge and return its bridge data.
        Uses caching for performance.
        """
        # Strip collection if present to get key
        if "/" in entity_key:
            entity_key = entity_key.split("/")[-1]

        if entity_key in self._bridge_cache:
            return self._bridge_cache[entity_key]
        
        aql = """
        FOR b IN @@bridge_col
          FILTER b.entity_key == @entity_key
          FILTER b.algorithm == @algorithm
          RETURN b
        """
        try:
            result = list(self.db.aql.execute(
                aql,
                bind_vars={
                    "@bridge_col": self.bridge_col,
                    "entity_key": entity_key,
                    "algorithm": self.algorithm,
                }
            ))
            bridge_data = result[0] if result else None
        except Exception:
            # Fallback if collection doesn't exist yet
            bridge_data = None

        self._bridge_cache[entity_key] = bridge_data
        return bridge_data

    def get_entity_community(self, entity_id: str) -> Optional[str]:
        """Get the community ID for an entity."""
        # For ArangoCommunityAccessor, we might know the community if mode is 'mapping'
        # But we should check the mapping collection to be sure (or if it's a bridge to another community)
        
        # If we are in 'mapping' mode, we can query membership collection
        if self.community_mode == "mapping":
            aql = f"""
            FOR m IN {self.membership_col}
              FILTER m.{self.memb_ent_field} == @entity_id
              // We don't filter by algorithm here usually, but if needed we can
              RETURN m.{self.memb_com_field}
            """
            try:
                result = list(self.db.aql.execute(aql, bind_vars={"entity_id": entity_id}))
                return result[0] if result else None
            except Exception:
                return None
        return None

    def get_affinity(self, community_a: str, community_b: str) -> float:
        """
        Get the affinity score between two communities.
        Returns 0.0 if no affinity data exists.
        """
        if not community_a or not community_b:
            return 0.0
            
        cache_key = f"{min(community_a, community_b)}_{max(community_a, community_b)}"
        
        if cache_key in self._affinity_cache:
            return self._affinity_cache[cache_key]
        
        aql = """
        FOR a IN @@affinity_col
          FILTER a.algorithm == @algorithm
          FILTER (a.community_a == @comm_a AND a.community_b == @comm_b)
              OR (a.community_a == @comm_b AND a.community_b == @comm_a)
          RETURN a.affinity_score
        """
        try:
            result = list(self.db.aql.execute(
                aql,
                bind_vars={
                    "@affinity_col": self.affinity_col,
                    "algorithm": self.algorithm,
                    "comm_a": community_a,
                    "comm_b": community_b,
                }
            ))
            affinity = result[0] if result else 0.0
        except Exception:
            affinity = 0.0
            
        self._affinity_cache[cache_key] = affinity
        return affinity

    def clear_bridge_cache(self):
        """Clear bridge/affinity caches."""
        self._bridge_cache.clear()
        self._affinity_cache.clear()

    # ════════════════════════════════════════════════════════════════
    # DISCOVERY ENTRY POINTS (for autonomous insight discovery)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def get_top_entities_in_community(
        db,
        community_id: str,
        membership_collection: str = "EntityCommunities",
        membership_entity_field: str = "entity_id",
        membership_community_field: str = "community_id",
        edges_collection: str = "ExtractedRelationships",
        limit: int = 20,
    ) -> List[dict]:
        """
        Get top entities by degree WITHIN a specific community.
        Essential for autonomous discovery - provides high-value seed nodes.
        
        Returns:
            List of {entity: str, degree: int}
        """
        aql = """
        LET community_entities = (
            FOR m IN @@membership
                FILTER m[@m_com] == @cid
                RETURN m[@m_ent]
        )
        FOR e IN @@edges
            FILTER e._from IN community_entities
            COLLECT entity = e._from WITH COUNT INTO degree
            SORT degree DESC
            LIMIT @limit
            RETURN { entity: entity, degree: degree }
        """
        return list(db.aql.execute(aql, bind_vars={
            "@membership": membership_collection,
            "@edges": edges_collection,
            "m_ent": membership_entity_field,
            "m_com": membership_community_field,
            "cid": community_id,
            "limit": limit,
        }))

    @staticmethod
    def get_recent_entities(
        db,
        since: str,  # ISO timestamp
        community_id: Optional[str] = None,
        nodes_collection: str = "ExtractedEntities",
        membership_collection: str = "EntityCommunities",
        membership_entity_field: str = "entity_id",
        membership_community_field: str = "community_id",
        created_at_property: str = "created_at",
        updated_at_property: str = "updated_at",
        limit: int = 100,
    ) -> List[dict]:
        """
        Get entities created or updated since a timestamp.
        Critical for daily discovery - "what's new since yesterday?"
        
        Args:
            since: ISO timestamp (e.g., "2026-01-11T00:00:00Z")
            community_id: Optional community filter
            
        Returns:
            List of {entity: str, created_at: str, type: str}
        """
        bind: Dict[str, Any] = {
            "since": since,
            "limit": limit,
            "created_prop": created_at_property,
            "updated_prop": updated_at_property,
        }
        
        community_filter = ""
        if community_id:
            community_filter = """
            LET community_entities = (
                FOR m IN @@membership
                    FILTER m[@m_com] == @cid
                    RETURN m[@m_ent]
            )
            FILTER e._id IN community_entities
            """
            bind["@membership"] = membership_collection
            bind["m_ent"] = membership_entity_field
            bind["m_com"] = membership_community_field
            bind["cid"] = community_id
        
        aql = f"""
        FOR e IN {nodes_collection}
            FILTER (HAS(e, @created_prop) AND e[@created_prop] >= @since)
                OR (HAS(e, @updated_prop) AND e[@updated_prop] >= @since)
            {community_filter}
            SORT HAS(e, @created_prop) ? e[@created_prop] : e[@updated_prop] DESC
            LIMIT @limit
            RETURN {{
                entity: e._id,
                created_at: HAS(e, @created_prop) ? e[@created_prop] : null,
                updated_at: HAS(e, @updated_prop) ? e[@updated_prop] : null,
                type: e.type,
                name: e.name
            }}
        """
        return list(db.aql.execute(aql, bind_vars=bind))

    @staticmethod
    def search_entities(
        db,
        query: str,
        community_id: Optional[str] = None,
        nodes_collection: str = "ExtractedEntities",
        membership_collection: str = "EntityCommunities",
        membership_entity_field: str = "entity_id",
        membership_community_field: str = "community_id",
        search_fields: List[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """
        Text search for entities matching query.
        Uses LIKE for simple text matching (can be upgraded to ArangoSearch).
        
        Args:
            query: Search string
            search_fields: Fields to search in (default: ["name", "description"])
            
        Returns:
            List of {entity: str, name: str, type: str, matched_field: str}
        """
        if search_fields is None:
            search_fields = ["name", "description"]
        
        bind: Dict[str, Any] = {
            "query": f"%{query.lower()}%",
            "limit": limit,
        }
        
        # Build search conditions
        search_conditions = []
        for field in search_fields:
            search_conditions.append(f"LOWER(e.{field}) LIKE @query")
        search_clause = " OR ".join(search_conditions)
        
        community_filter = ""
        if community_id:
            community_filter = """
            LET community_entities = (
                FOR m IN @@membership
                    FILTER m[@m_com] == @cid
                    RETURN m[@m_ent]
            )
            FILTER e._id IN community_entities
            """
            bind["@membership"] = membership_collection
            bind["m_ent"] = membership_entity_field
            bind["m_com"] = membership_community_field
            bind["cid"] = community_id
        
        aql = f"""
        FOR e IN {nodes_collection}
            FILTER {search_clause}
            {community_filter}
            LIMIT @limit
            RETURN {{
                entity: e._id,
                name: e.name,
                type: e.type,
                description: e.description
            }}
        """
        return list(db.aql.execute(aql, bind_vars=bind))

    # ════════════════════════════════════════════════════════════════
    # CONTENT HYDRATION (for agent reasoning)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def get_document_content(
        db,
        doc_id: str,
        text_collection: str = "TextBlocks",
        table_collection: str = "Tables",
        image_collection: str = "Images",
        document_collection: str = "Documents",
    ) -> Optional[dict]:
        """
        Fetch content from any document collection by ID.
        Essential for agent reasoning - converts graph IDs to actual content.
        
        Args:
            doc_id: Document ID in format "CollectionName/key"
            
        Returns:
            Dict with type-specific content, or None if not found
        """
        try:
            collection, key = doc_id.split("/", 1)
        except ValueError:
            return None
        
        if collection == text_collection:
            aql = f"""
            FOR tb IN {text_collection}
                FILTER tb._id == @doc_id
                RETURN {{
                    type: "text",
                    text: tb.text,
                    document_id: tb.document_id,
                    page: tb.page,
                    char_span: tb.char_span,
                    metadata: tb.metadata
                }}
            """
        elif collection == table_collection:
            aql = f"""
            FOR t IN {table_collection}
                FILTER t._id == @doc_id
                RETURN {{
                    type: "table",
                    headers: t.headers,
                    rows: t.rows,
                    caption: t.caption,
                    document_id: t.document_id,
                    page: t.page,
                    metadata: t.metadata
                }}
            """
        elif collection == image_collection:
            aql = f"""
            FOR img IN {image_collection}
                FILTER img._id == @doc_id
                RETURN {{
                    type: "image",
                    caption: img.caption,
                    ocr_text: img.ocr_text,
                    url: img.storage_url,
                    document_id: img.document_id,
                    page: img.page,
                    metadata: img.metadata
                }}
            """
        elif collection == document_collection:
            aql = f"""
            FOR d IN {document_collection}
                FILTER d._id == @doc_id
                RETURN {{
                    type: "document",
                    filename: d.filename,
                    content: d.content,
                    metadata: d.metadata
                }}
            """
        else:
            return None
        
        result = list(db.aql.execute(aql, bind_vars={"doc_id": doc_id}))
        return result[0] if result else None

    @staticmethod
    def get_entity_sources(
        db,
        entity_id: str,
        extracted_from_collection: str = "EXTRACTED_FROM",
        max_sources: int = 10,
    ) -> List[dict]:
        """
        Get all source documents/blocks for an entity via EXTRACTED_FROM edges.
        Critical for evidence gathering - shows WHERE an entity was mentioned.
        
        Args:
            entity_id: Entity ID (e.g., "ExtractedEntities/ent_123")
            max_sources: Limit number of sources returned
            
        Returns:
            List of {source_id, source_type, content, char_span, confidence, metadata}
        """
        aql = f"""
        FOR edge IN {extracted_from_collection}
            FILTER edge._from == @entity_id
            LIMIT @max_sources
            LET source = DOCUMENT(edge._to)
            LET collection = PARSE_IDENTIFIER(edge._to).collection
            RETURN {{
                source_id: edge._to,
                source_type: collection,
                char_span: edge.char_span,
                extraction_confidence: edge.extraction_confidence,
                content: CASE
                    WHEN collection == "TextBlocks" THEN source.text
                    WHEN collection == "Tables" THEN {{ headers: source.headers, rows: source.rows }}
                    WHEN collection == "Images" THEN {{ caption: source.caption, ocr_text: source.ocr_text }}
                    WHEN collection == "Documents" THEN SUBSTRING(source.content, 0, 500)
                    ELSE null
                END,
                metadata: {{
                    page: source.page,
                    document_id: source.document_id,
                    filename: source.filename
                }}
            }}
        """
        return list(db.aql.execute(aql, bind_vars={
            "entity_id": entity_id,
            "max_sources": max_sources,
        }))

    @staticmethod
    def search_content(
        db,
        query: str,
        community_id: Optional[str] = None,
        content_types: List[str] = None,
        text_collection: str = "TextBlocks",
        table_collection: str = "Tables",
        image_collection: str = "Images",
        membership_collection: str = "EntityCommunities",
        extracted_from_collection: str = "EXTRACTED_FROM",
        limit: int = 10,
    ) -> List[dict]:
        """
        Semantic/text search across content collections.
        Uses simple LIKE matching (can be upgraded to ArangoSearch/vectors).
        
        Args:
            query: Search string
            content_types: Collections to search (default: ["TextBlocks", "Tables", "Images"])
            community_id: Optional filter to content linked to community entities
            
        Returns:
            List of {source_id, source_type, content, score, metadata}
        """
        if content_types is None:
            content_types = [text_collection, table_collection, image_collection]
        
        bind: Dict[str, Any] = {
            "query": f"%{query.lower()}%",
            "limit": limit,
        }
        
        results = []
        
        # Search TextBlocks
        if text_collection in content_types:
            aql_text = f"""
            FOR tb IN {text_collection}
                FILTER LOWER(tb.text) LIKE @query
                LIMIT @limit
                RETURN {{
                    source_id: tb._id,
                    source_type: "TextBlocks",
                    content: tb.text,
                    score: 1.0,
                    metadata: {{
                        document_id: tb.document_id,
                        page: tb.page
                    }}
                }}
            """
            results.extend(list(db.aql.execute(aql_text, bind_vars=bind)))
        
        # Search Tables (caption)
        if table_collection in content_types:
            aql_table = f"""
            FOR t IN {table_collection}
                FILTER LOWER(t.caption) LIKE @query
                LIMIT @limit
                RETURN {{
                    source_id: t._id,
                    source_type: "Tables",
                    content: {{ headers: t.headers, rows: t.rows, caption: t.caption }},
                    score: 1.0,
                    metadata: {{
                        document_id: t.document_id,
                        page: t.page
                    }}
                }}
            """
            results.extend(list(db.aql.execute(aql_table, bind_vars=bind)))
        
        # Search Images (OCR text)
        if image_collection in content_types:
            aql_image = f"""
            FOR img IN {image_collection}
                FILTER LOWER(img.ocr_text) LIKE @query OR LOWER(img.caption) LIKE @query
                LIMIT @limit
                RETURN {{
                    source_id: img._id,
                    source_type: "Images",
                    content: {{ caption: img.caption, ocr_text: img.ocr_text }},
                    score: 1.0,
                    metadata: {{
                        document_id: img.document_id,
                        page: img.page
                    }}
                }}
            """
            results.extend(list(db.aql.execute(aql_image, bind_vars=bind)))
        
        return results[:limit]

    # --------------------------
    # Internal neighbor routine
    # --------------------------
    def _iter_neighbors(self, node: NodeId, *, direction: str, rich: bool) -> Iterable[EdgeView]:
        assert direction in ("OUTBOUND", "INBOUND")

        bind: Dict[str, Any] = {
            "node": node,
            "rel_prop": self.rel_prop,
            "w_prop": self.w_prop,
            "priors_map": self.type_priors,
        }
        
        # Only add community ID if we're filtering by it
        if self.community_mode != "none":
            bind["cid"] = self._cid
        
        # Bind parameters are added only when referenced to avoid AQL 1552 errors

        hint = ""
        if direction == "OUTBOUND" and self.outbound_index_hint:
            hint = "OPTIONS { indexHint: @idx, forceIndexHint: true }"
            bind["idx"] = self.outbound_index_hint
        elif direction == "INBOUND" and self.inbound_index_hint:
            hint = "OPTIONS { indexHint: @idx, forceIndexHint: true }"
            bind["idx"] = self.inbound_index_hint

        filters: List[str] = []

        # Community filter
        if self.community_mode == "property":
            filters.append(f"v.{self.community_prop} == @cid")
        elif self.community_mode == "mapping":
            bind.update({"@mcol": self.membership_col, "m_ent": self.memb_ent_field, "m_com": self.memb_com_field})
            filters.append("""
              FIRST(
                FOR m IN @@mcol
                  FILTER m[@m_com] == @cid AND m[@m_ent] == v._id
                  LIMIT 1
                  RETURN 1
              )
            """)
        # else community_mode == "none" - no filtering

        # Relation / neighbor type filters
        if self.allowed_relations:
            bind["allowed_relations"] = self.allowed_relations
            filters.append("e[@rel_prop] IN @allowed_relations")
        if self.disallowed_relations:
            bind["disallowed_relations"] = self.disallowed_relations
            filters.append("!(e[@rel_prop] IN @disallowed_relations)")
        if self.allowed_neighbor_types:
            bind["allowed_neighbor_types"] = self.allowed_neighbor_types
            filters.append(f"v.{self.node_type_prop} IN @allowed_neighbor_types")

        # Time window filter on edge timestamp
        if self.time_window and self.ts_prop:
            bind["start_ts"], bind["end_ts"] = self.time_window
            bind["ts_prop"] = self.ts_prop
            filters.append("HAS(e, @ts_prop) AND e[@ts_prop] >= @start_ts AND e[@ts_prop] <= @end_ts")

        # Current-only validity wrt as_of
        if self.current_only and self.as_of:
            bind["as_of"] = self.as_of
            vf_prop = self.edge_valid_from_prop or "valid_from"
            vt_prop = self.edge_valid_to_prop or "valid_to"
            filters.append(
                f"( (HAS(e, '{vf_prop}') ? e['{vf_prop}'] <= @as_of : true) "
                f"AND (HAS(e, '{vt_prop}') ? (e['{vt_prop}'] == null OR e['{vt_prop}'] >= @as_of) : true) )"
            )

        # Optional status guard
        status_guard = ""
        if self.edge_status_prop:
            status_guard = "LET _status = e[@status_prop]"
            bind["status_prop"] = self.edge_status_prop

        # Recency decay: 2^(- age_days / half_life)
        recency_clause = "1.0"
        if self.recency_half_life_days is not None and self.as_of and self.ts_prop:
            bind["half_life"] = float(self.recency_half_life_days)
            bind["as_of"] = self.as_of
            bind["ts_prop"] = self.ts_prop
            recency_clause = "POW(2, -1 * DATE_DIFF(@as_of, e[@ts_prop], 'days') / @half_life)"

        # Base weight
        weight_clause = "(HAS(e, @w_prop) && IS_NUMBER(e[@w_prop]) ? e[@w_prop] : 1.0)"

        # Confidence fusion (usually disabled; you do it in engine)
        conf_clause = "1.0"
        if self.fuse_edge_confidence:
            bind.update({
                "raw_c": self.edge_raw_conf_prop,
                "npll": self.edge_npll_post_prop,
                "calib": self.edge_calibration_prop,
                "miss_prior": float(self.missing_confidence_prior),
            })
            conf_clause = (
                "( (HAS(e, @raw_c)  && IS_NUMBER(e[@raw_c])  ? e[@raw_c]  : @miss_prior) * "
                "  (HAS(e, @npll)   && IS_NUMBER(e[@npll])   ? e[@npll]   : @miss_prior) * "
                "  (HAS(e, @calib)  && IS_NUMBER(e[@calib])  ? e[@calib]  : @miss_prior) )"
            )

        filters_str = " && ".join(filters) if filters else "true"

        # Build the src edges clause safely
        src_edges_clause = (
            f"""
            FOR p IN {self.prov_edges_col}
              FILTER p._from IN [e._from, e._to]
              RETURN p._to
            """
            if self.prov_edges_col else "[]"
        )

        aql = f"""
        LET priors = @priors_map
        FOR v, e IN 1..1 {direction} @node {self.edges_col}
          {hint}
          FILTER {filters_str}
          {status_guard}
          LET _rel = e[@rel_prop]
          LET _prior = TO_NUMBER(NOT_NULL(priors[_rel], 1.0))
          LET _base_w = {weight_clause}
          LET _rec    = {recency_clause}
          LET _conf   = {conf_clause}
          LET _w_eff  = TO_NUMBER(_base_w) * TO_NUMBER(_prior) * TO_NUMBER(_rec) * TO_NUMBER(_conf)

          LET _vf = {f"e['{self.edge_valid_from_prop}']" if self.edge_valid_from_prop else 'null'}
          LET _vt = {f"e['{self.edge_valid_to_prop}']" if self.edge_valid_to_prop else 'null'}
          LET _status2 = {f"e['{self.edge_status_prop}']" if self.edge_status_prop else 'null'}

          // Provenance: inline fields + EXTRACTED_FROM for both endpoints
          LET _src_inline_candidates = [{", ".join([f"e['{f}']" for f in self.edge_prov_fields])}]
          LET _src_inline = (
            FOR x IN _src_inline_candidates
              FILTER x != null
              RETURN x
          )
          LET _src_edges = (
            {src_edges_clause}
          )
          LET _sources = UNIQUE(APPEND(_src_inline, _src_edges))

          RETURN {{
            v_id: v._id,
            rel: _rel,
            weight: _w_eff,
            edge_id: e._id,
            valid_from: _vf,
            valid_to: _vt,
            status: _status2,
            raw_confidence: {f"e['{self.edge_raw_conf_prop}']" if self.edge_raw_conf_prop else 'null'},
            npll_posterior: {f"e['{self.edge_npll_post_prop}']" if self.edge_npll_post_prop else 'null'},
            calibration: {f"e['{self.edge_calibration_prop}']" if self.edge_calibration_prop else 'null'},
            sources: _sources
          }}
        """

        cursor = self.db.aql.execute(
            aql,
            bind_vars=bind,
            batch_size=self.aql_batch_size or 1000,
            stream=self.aql_stream if self.aql_stream is not None else True,
            ttl=120,  # 2 minute timeout for long queries
            optimizer_rules=["+use-indexes"]  # Force index usage
        )
        for d in cursor:
            if rich:
                yield EdgeView(
                    neighbor_id=d["v_id"],
                    relation=d["rel"],
                    weight=float(d["weight"]),
                    edge_id=d["edge_id"],
                    valid_from=d.get("valid_from"),
                    valid_to=d.get("valid_to"),
                    status=d.get("status"),
                    raw_confidence=d.get("raw_confidence"),
                    npll_posterior=d.get("npll_posterior"),
                    calibration=d.get("calibration"),
                    sources=d.get("sources") or [],
                )
            else:
                yield d["v_id"], d["rel"], float(d["weight"])


class GlobalGraphAccessor(GraphAccessor):
    """
    Cross-community graph accessor using pre-computed bridge entities.
    
    This accessor enables intelligent traversal across community boundaries
    by leveraging the BridgeEntities and CommunityAffinity collections
    created during community detection.
    
    Key features:
    - Uses bridge entities to efficiently cross community boundaries
    - Scores cross-community paths using affinity scores
    - Mission-aware: can weight community crossings based on context
    - Maintains all ArangoCommunityAccessor features
    """

    def __init__(
        self,
        db,
        algorithm: str = "leiden",
        # Base accessor settings
        nodes_collection: str = "ExtractedEntities",
        edges_collection: str = "ExtractedRelationships",
        relation_property: str = "relationship",
        weight_property: str = "weight",
        # Bridge collections
        bridge_collection: str = "BridgeEntities",
        affinity_collection: str = "CommunityAffinity",
        membership_collection: str = "EntityCommunities",
        # Cross-community scoring
        cross_community_bonus: float = 1.5,  # Boost for cross-community edges (often valuable)
        min_affinity_threshold: float = 0.0,  # Minimum affinity to allow crossing
        # Performance
        aql_batch_size: int = 1000,
        aql_stream: bool = True,
    ):
        self.db = db
        self.algorithm = algorithm
        
        self.nodes_col = nodes_collection
        self.edges_col = edges_collection
        self.rel_prop = relation_property
        self.w_prop = weight_property
        
        self.bridge_col = bridge_collection
        self.affinity_col = affinity_collection
        self.membership_col = membership_collection
        
        self.cross_community_bonus = cross_community_bonus
        self.min_affinity_threshold = min_affinity_threshold
        
        self.aql_batch_size = aql_batch_size
        self.aql_stream = aql_stream
        
        # Cache for bridge status and affinities
        self._bridge_cache: Dict[str, Optional[dict]] = {}
        self._affinity_cache: Dict[str, float] = {}

    # --------------------------
    # Core traversal API
    # --------------------------
    
    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Iterate outbound edges, scoring cross-community edges appropriately."""
        for ev in self._iter_neighbors_global(node, direction="OUTBOUND"):
            yield ev.neighbor_id, ev.relation, ev.weight

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Iterate inbound edges, scoring cross-community edges appropriately."""
        for ev in self._iter_neighbors_global(node, direction="INBOUND"):
            yield ev.neighbor_id, ev.relation, ev.weight

    def iter_out_rich(self, node: NodeId) -> Iterable[EdgeView]:
        """Rich outbound edges with cross-community metadata."""
        yield from self._iter_neighbors_global(node, direction="OUTBOUND")

    def iter_in_rich(self, node: NodeId) -> Iterable[EdgeView]:
        """Rich inbound edges with cross-community metadata."""
        yield from self._iter_neighbors_global(node, direction="INBOUND")

    def nodes(self) -> Iterable[NodeId]:
        """Return all nodes (no community restriction)."""
        aql = f"FOR v IN {self.nodes_col} RETURN v._id"
        cursor = self.db.aql.execute(aql, batch_size=self.aql_batch_size, stream=self.aql_stream)
        for vid in cursor:
            yield vid

    def degree(self, node: NodeId) -> int:
        """Out-degree of a node."""
        aql = f"""
        RETURN LENGTH(
          FOR e IN {self.edges_col}
            FILTER e._from == @node
            RETURN 1
        )
        """
        cur = self.db.aql.execute(aql, bind_vars={"node": node})
        return int(list(cur)[0] or 0)

    # --------------------------
    # Bridge-aware methods
    # --------------------------

    def is_bridge(self, entity_key: str) -> Optional[dict]:
        """
        Check if an entity is a bridge and return its bridge data.
        Uses caching for performance.
        """
        if entity_key in self._bridge_cache:
            return self._bridge_cache[entity_key]
        
        aql = """
        FOR b IN @@bridge_col
          FILTER b.entity_key == @entity_key
          FILTER b.algorithm == @algorithm
          RETURN b
        """
        result = list(self.db.aql.execute(
            aql,
            bind_vars={
                "@bridge_col": self.bridge_col,
                "entity_key": entity_key,
                "algorithm": self.algorithm,
            }
        ))
        
        bridge_data = result[0] if result else None
        self._bridge_cache[entity_key] = bridge_data
        return bridge_data

    def get_entity_community(self, entity_id: str) -> Optional[str]:
        """Get the community ID for an entity."""
        aql = """
        FOR m IN @@membership_col
          FILTER m.entity_id == @entity_id
          FILTER m.algorithm == @algorithm
          RETURN m.community_id
        """
        result = list(self.db.aql.execute(
            aql,
            bind_vars={
                "@membership_col": self.membership_col,
                "entity_id": entity_id,
                "algorithm": self.algorithm,
            }
        ))
        return result[0] if result else None

    def get_affinity(self, community_a: str, community_b: str) -> float:
        """
        Get the affinity score between two communities.
        Returns 0.0 if no affinity data exists.
        """
        cache_key = f"{min(community_a, community_b)}_{max(community_a, community_b)}"
        
        if cache_key in self._affinity_cache:
            return self._affinity_cache[cache_key]
        
        aql = """
        FOR a IN @@affinity_col
          FILTER a.algorithm == @algorithm
          FILTER (a.community_a == @comm_a AND a.community_b == @comm_b)
              OR (a.community_a == @comm_b AND a.community_b == @comm_a)
          RETURN a.affinity_score
        """
        result = list(self.db.aql.execute(
            aql,
            bind_vars={
                "@affinity_col": self.affinity_col,
                "algorithm": self.algorithm,
                "comm_a": community_a,
                "comm_b": community_b,
            }
        ))
        
        affinity = result[0] if result else 0.0
        self._affinity_cache[cache_key] = affinity
        return affinity

    def get_bridges_from_community(self, community_id: str, min_strength: int = 1) -> List[dict]:
        """Get all bridge entities from a specific community."""
        aql = """
        FOR b IN @@bridge_col
          FILTER b.algorithm == @algorithm
          FILTER b.home_community == @community_id
          FILTER b.bridge_strength >= @min_strength
          SORT b.bridge_strength DESC
          RETURN b
        """
        return list(self.db.aql.execute(
            aql,
            bind_vars={
                "@bridge_col": self.bridge_col,
                "algorithm": self.algorithm,
                "community_id": community_id,
                "min_strength": min_strength,
            }
        ))

    def get_top_bridges(self, limit: int = 20) -> List[dict]:
        """Get the top bridge entities by bridge strength."""
        aql = """
        FOR b IN @@bridge_col
          FILTER b.algorithm == @algorithm
          SORT b.bridge_strength DESC
          LIMIT @limit
          RETURN b
        """
        return list(self.db.aql.execute(
            aql,
            bind_vars={
                "@bridge_col": self.bridge_col,
                "algorithm": self.algorithm,
                "limit": limit,
            }
        ))

    def get_strongest_affinities(self, limit: int = 20) -> List[dict]:
        """Get the strongest inter-community affinities."""
        aql = """
        FOR a IN @@affinity_col
          FILTER a.algorithm == @algorithm
          SORT a.affinity_score DESC
          LIMIT @limit
          RETURN a
        """
        return list(self.db.aql.execute(
            aql,
            bind_vars={
                "@affinity_col": self.affinity_col,
                "algorithm": self.algorithm,
                "limit": limit,
            }
        ))

    # --------------------------
    # Cross-community traversal
    # --------------------------

    def _iter_neighbors_global(self, node: NodeId, direction: str) -> Iterable[EdgeView]:
        """
        Iterate neighbors with cross-community awareness.
        
        - Gets all neighbors (no community restriction)
        - Detects cross-community edges
        - Applies bonus/penalty based on affinity
        """
        assert direction in ("OUTBOUND", "INBOUND")
        
        # Get source node's community
        source_community = self.get_entity_community(node)
        
        # Get all neighbors
        aql = f"""
        FOR v, e IN 1..1 {direction} @node {self.edges_col}
          LET rel = e[@rel_prop]
          LET base_weight = HAS(e, @w_prop) && IS_NUMBER(e[@w_prop]) ? e[@w_prop] : 1.0
          RETURN {{
            v_id: v._id,
            v_key: v._key,
            rel: rel,
            base_weight: base_weight,
            edge_id: e._id,
            sources: []
          }}
        """
        
        cursor = self.db.aql.execute(
            aql,
            bind_vars={
                "node": node,
                "rel_prop": self.rel_prop,
                "w_prop": self.w_prop,
            },
            batch_size=self.aql_batch_size,
            stream=self.aql_stream,
        )
        
        for d in cursor:
            neighbor_id = d["v_id"]
            base_weight = float(d["base_weight"])
            
            # Check if this is a cross-community edge
            neighbor_community = self.get_entity_community(neighbor_id)
            
            weight = base_weight
            is_cross_community = False
            
            if source_community and neighbor_community and source_community != neighbor_community:
                is_cross_community = True
                
                # Get affinity between communities
                affinity = self.get_affinity(source_community, neighbor_community)
                
                # Apply cross-community scoring
                if affinity >= self.min_affinity_threshold:
                    # Bonus for crossing to well-connected communities
                    weight = base_weight * self.cross_community_bonus * (1 + affinity)
                else:
                    # Penalty for crossing to poorly-connected communities
                    weight = base_weight * 0.5
            
            yield EdgeView(
                neighbor_id=neighbor_id,
                relation=d["rel"],
                weight=weight,
                edge_id=d["edge_id"],
                valid_from=None,
                valid_to=None,
                status="cross_community" if is_cross_community else "same_community",
                raw_confidence=None,
                npll_posterior=None,
                calibration=None,
                sources=d.get("sources") or [],
            )

    # --------------------------
    # Mission-aware scoring
    # --------------------------

    def score_community_crossing(
        self,
        from_community: str,
        to_community: str,
        mission: Optional[str] = None
    ) -> float:
        """
        Score a community crossing based on mission context.
        
        Args:
            from_community: Source community
            to_community: Target community
            mission: Optional mission context (e.g., "fraud_detection", "patient_care")
            
        Returns:
            Score multiplier for the crossing (>1 = valuable, <1 = not valuable)
        """
        base_affinity = self.get_affinity(from_community, to_community)
        
        if not mission:
            return 1.0 + base_affinity
        
        # Mission-specific scoring (customize based on your domain)
        mission_lower = mission.lower()
        
        # Example: fraud detection values Claims -> Clinical crossings
        if "fraud" in mission_lower:
            # This would need actual community type detection
            # For now, just boost high-affinity crossings
            return (1.0 + base_affinity) * 1.5
        
        # Example: patient care values Clinical -> Lab crossings
        if "patient" in mission_lower or "clinical" in mission_lower:
            return (1.0 + base_affinity) * 1.3
        
        return 1.0 + base_affinity

    def clear_cache(self):
        """Clear internal caches (useful after data updates)."""
        self._bridge_cache.clear()
        self._affinity_cache.clear()
