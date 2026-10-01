from __future__ import annotations
from typing import Iterable, Tuple, Optional, List, Dict, Any, NamedTuple

from .adapters import GraphAccessor, NodeId, RelId
from .evidence import clean_evidence


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
    sources: List[str]             # source IDs from configured inline/provenance mappings
    assertion: Dict[str, Any]
    timestamp: Optional[str] = None


def edge_record(node: NodeId, edge: EdgeView) -> Dict[str, Any]:
    from .evidence import exclude_vectors
    assertion, excluded = exclude_vectors(edge.assertion, "provenance.assertion")
    return {"_id": edge.edge_id, "u": node, "rel": edge.relation,
            "v": edge.neighbor_id, "weight": edge.weight,
            "created_at": edge.timestamp, "valid_from": edge.valid_from,
            "valid_to": edge.valid_to, "status": edge.status,
            "source_confidence": edge.raw_confidence,
            "npll_posterior": edge.npll_posterior, "calibration": edge.calibration,
            "provenance": {"assertion": assertion, "sources": edge.sources},
            "excluded_vector_fields": excluded}



class ArangoCommunityAccessor(GraphAccessor):
    """
    Arango-backed GraphAccessor for a single community.

    Every application collection and schema field is supplied explicitly by the
    caller or a backend-owned graph configuration.

    Structural weight only by default:
        w_struct = base_weight * type_prior(relation) * recency_decay
    (Set fuse_edge_confidence=True to multiply raw_confidence * npll_posterior * calibration in-adapter.)
    """

    def __init__(
        self,
        db,
        community_id: str,
        # Collections
        nodes_collection: str,
        edges_collection: str,
        # Core field names
        relation_property: str,
        weight_property: Optional[str] = None,
        node_type_property: Optional[str] = None,
        # Time fields
        edge_timestamp_property: Optional[str] = None,
        edge_valid_from_property: Optional[str] = None,
        edge_valid_to_property: Optional[str] = None,
        edge_status_property: Optional[str] = None,
        allowed_edge_statuses: Optional[List[str]] = None,
        # Community scoping (mapping mode by default)
        community_mode: str = "none",  # "none" | "mapping" | "property"
        community_property: Optional[str] = None,
        membership_collection: Optional[str] = None,
        membership_entity_field: Optional[str] = None,
        membership_community_field: Optional[str] = None,
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
        edge_provenance_fields: Optional[List[str]] = None,
        provenance_edge_collection: Optional[str] = None,
        provenance_target_collections: Optional[List[str]] = None,
        # Confidence fusion (usually False; you do NPLL in engine)
        fuse_edge_confidence: bool = False,
        missing_confidence_prior: float = 1.0,
        edge_raw_confidence_property: Optional[str] = None,
        edge_npll_posterior_property: Optional[str] = None,
        edge_calibration_property: Optional[str] = None,
        # Performance
        aql_batch_size: int = 1000,
        aql_stream: bool = True,
        outbound_index_hint: Optional[str] = None,  # e.g. "edges_from_rel_ts"
        inbound_index_hint: Optional[str] = None,   # e.g. "edges_to_rel_ts"
        # Bridge / GNN integration
        bridge_collection: Optional[str] = None,
        affinity_collection: Optional[str] = None,
        algorithm: Optional[str] = None,
        membership_algorithm_field: Optional[str] = None,
        bridge_entity_field: Optional[str] = None,
        bridge_strength_field: Optional[str] = None,
        bridge_community_field: Optional[str] = None,
        bridge_algorithm_field: Optional[str] = None,
        affinity_from_field: Optional[str] = None,
        affinity_to_field: Optional[str] = None,
        affinity_score_field: Optional[str] = None,
        affinity_algorithm_field: Optional[str] = None,
    ):
        self.db = db
        if community_mode not in {"none", "mapping", "property"}:
            raise ValueError("community_mode must be 'none', 'mapping', or 'property'")
        if community_mode == "property" and not community_property:
            raise ValueError("property community mode requires community_property")
        if community_mode == "mapping" and not all((
            membership_collection,
            membership_entity_field,
            membership_community_field,
        )):
            raise ValueError("mapping community mode requires membership mapping fields")
        for label, values in (
            ("bridge", (bridge_collection, bridge_entity_field, bridge_strength_field,
                        bridge_community_field)),
            ("affinity", (affinity_collection, affinity_from_field, affinity_to_field,
                          affinity_score_field)),
        ):
            if any(v is not None for v in values) and not all(
                isinstance(v, str) and v for v in values
            ):
                raise ValueError(f"{label} access requires complete mapping fields")
        if any((membership_algorithm_field, bridge_algorithm_field, affinity_algorithm_field)) and not algorithm:
            raise ValueError("algorithm field mappings require algorithm")
        self._cid = community_id
        self.bridge_col = bridge_collection
        self.affinity_col = affinity_collection
        self.algorithm = algorithm
        self.membership_algorithm_field = membership_algorithm_field
        self.bridge_entity_field = bridge_entity_field
        self.bridge_strength_field = bridge_strength_field
        self.bridge_community_field = bridge_community_field
        self.bridge_algorithm_field = bridge_algorithm_field
        self.affinity_from_field = affinity_from_field
        self.affinity_to_field = affinity_to_field
        self.affinity_score_field = affinity_score_field
        self.affinity_algorithm_field = affinity_algorithm_field
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
        self.allowed_edge_statuses = allowed_edge_statuses
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

        self.edge_prov_fields = edge_provenance_fields or []
        self.prov_edges_col = provenance_edge_collection
        self.prov_target_cols = provenance_target_collections or []

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

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        """Arango retrieval accepts full document IDs without remapping."""
        return seeds

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        for ev in self._iter_neighbors(node, direction="INBOUND", rich=True):
            yield ev.neighbor_id, ev.relation, ev.weight

    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        """
        Return all node IDs in this community.
        - mapping mode: configured membership collection -> configured entity ID
        - property mode: filter configured nodes by configured property field
        - none mode: return all nodes
        """
        cid = community_id or self._cid
        if self.community_mode == "property":
            aql = f"""
            FOR v IN @@nodes
              FILTER v[@community_field] == @cid
              RETURN v._id
            """
            cursor = self.db.aql.execute(
                aql, bind_vars={"cid": cid, "@nodes": self.nodes_col, "community_field": self.community_prop}, batch_size=self.aql_batch_size, stream=self.aql_stream
            )
        elif self.community_mode == "mapping":
            bind = {"cid": cid, "@mcol": self.membership_col,
                    "m_ent": self.memb_ent_field, "m_com": self.memb_com_field}
            guard = self._algorithm_filter("m", self.membership_algorithm_field, bind)
            aql = f"""
            FOR m IN @@mcol
              FILTER m[@m_com] == @cid
              {guard}
              RETURN m[@m_ent]
            """
            cursor = self.db.aql.execute(
                aql,
                bind_vars=bind,
                batch_size=self.aql_batch_size,
                stream=self.aql_stream,
            )
        else:  # community_mode == "none"
            aql = f"""
            FOR v IN @@nodes
              RETURN v._id
            """
            cursor = self.db.aql.execute(
                aql, bind_vars={"@nodes": self.nodes_col}, batch_size=self.aql_batch_size, stream=self.aql_stream
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
          FOR e IN @@edges
            {hint_clause}
            FILTER e._from == @node
            RETURN 1
        )
        """
        bind = {"node": node, "@edges": self.edges_col}
        if self.outbound_index_hint:
            bind["idx"] = self.outbound_index_hint
        cur = self.db.aql.execute(aql, bind_vars=bind)
        return int(list(cur)[0] or 0)

    # --------------------------
    # Rich neighbor variants
    # --------------------------
    def iter_out_rich(self, node: NodeId) -> Iterable[EdgeView]:
        yield from self._iter_neighbors(node, direction="OUTBOUND", rich=True)

    def iter_out_edges(self, node: NodeId):
        for edge in self.iter_out_rich(node):
            yield edge_record(node, edge)

    def iter_in_rich(self, node: NodeId) -> Iterable[EdgeView]:
        yield from self._iter_neighbors(node, direction="INBOUND", rich=True)

    # --------------------------
    # Provenance helpers
    # --------------------------
    def get_edge_provenance(self, edge_id: str) -> List[str]:
        """
        Return provenance targets for a relationship edge:
          - inline fields (source_document_id, source_text_id)
          - configured provenance edges for either endpoint entity
        """
        # Build the provenance edges clause safely (avoid nested f-strings)
        prov_edges_clause = (
            f"""
            FOR p IN @@provenance_edges
              FILTER p._from IN [e._from, e._to]
              FILTER LENGTH(@provenance_targets) == 0 OR PARSE_IDENTIFIER(p._to).collection IN @provenance_targets
              RETURN p._to
            """
            if self.prov_edges_col else "[]"
        )

        aql = f"""
        LET e = DOCUMENT(@eid)
        LET inline_candidates = (FOR field IN @inline_fields RETURN e[field])
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
        bind = {"eid": edge_id, "inline_fields": self.edge_prov_fields}
        if self.prov_edges_col:
            bind.update({"@provenance_edges": self.prov_edges_col,
                         "provenance_targets": self.prov_target_cols})
        cur = self.db.aql.execute(aql, bind_vars=bind, batch_size=self.aql_batch_size, stream=self.aql_stream)
        out = list(cur)
        return out[0] if out else []

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        bind = {"id": node_id}
        if fields:
            aql = "LET d = DOCUMENT(@id) FILTER d != null RETURN KEEP(d, @fields)"
            bind["fields"] = list(dict.fromkeys(["_id", *fields]))
        else:
            aql = "RETURN DOCUMENT(@id)"
        rows = list(self.db.aql.execute(aql, bind_vars=bind))
        return clean_evidence((rows[0] or {}) if rows else {})

    @staticmethod
    def get_top_n_entities_by_degree(
        db,
        edges_collection: str,
        limit: Optional[int] = None,
        time_window: Optional[Tuple[str, str]] = None,
        time_property: Optional[str] = None,
    ) -> List[dict]:
        bind: Dict[str, Any] = {"@edges": edges_collection}
        where = ""
        if time_window:
            if time_property is None:
                raise ValueError("time_property is required with time_window")
            where = "FILTER HAS(e, @ts) AND e[@ts] >= @start_ts AND e[@ts] <= @end_ts"
            bind.update({"ts": time_property, "start_ts": time_window[0], "end_ts": time_window[1]})
        limit_clause = "LIMIT @lim" if limit else ""
        if limit:
            bind["lim"] = limit
        aql = f"""
        FOR e IN @@edges
          {where}
          COLLECT entity = e._from WITH COUNT INTO degree
          SORT degree DESC
          {limit_clause}
          RETURN {{ "entity": entity, "degree": degree }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    @staticmethod
    def get_entity_type_counts(
        db,
        nodes_collection: str,
        type_property: str,
    ) -> List[dict]:
        aql = f"""
        FOR doc IN @@nodes
          COLLECT t = doc[@type_field] WITH COUNT INTO c
          SORT c DESC
          RETURN {{ "type": t, "count": c }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars={"@nodes": nodes_collection, "type_field": type_property})))

    @staticmethod
    def get_relationship_type_counts(
        db,
        edges_collection: str,
        relation_property: str,
        time_window: Optional[Tuple[str, str]] = None,
        time_property: Optional[str] = None,
    ) -> List[dict]:
        bind: Dict[str, Any] = {"@edges": edges_collection, "rel_prop": relation_property}
        where = "FILTER HAS(rel, @rel_prop)"
        if time_window:
            if time_property is None:
                raise ValueError("time_property is required with time_window")
            where += " AND HAS(rel, @ts) AND rel[@ts] >= @start_ts AND rel[@ts] <= @end_ts"
            bind.update({"ts": time_property, "start_ts": time_window[0], "end_ts": time_window[1]})
        aql = f"""
        FOR rel IN @@edges
          {where}
          COLLECT t = rel[@rel_prop] WITH COUNT INTO c
          SORT c DESC
          RETURN {{ "type": t, "count": c }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    @staticmethod
    def get_community_summaries(
        db,
        communities_collection: str,
        community_id_property: str,
        summary_property: str,
        size_property: str,
        level_property: str,
        limit: Optional[int] = None,
        skip: int = 0,
        require_summary: bool = True
    ) -> List[dict]:
        filter_clause = (
            "FILTER HAS(c, @summary_property) AND c[@summary_property] != null "
            "AND c[@summary_property] != ''"
            if require_summary
            else "FILTER !HAS(c, @summary_property) OR c[@summary_property] == null "
            "OR c[@summary_property] == ''"
        )
        limit_clause = "LIMIT @skip, @limit" if limit is not None else ""
        bind: Dict[str, Any] = {
            "@communities": communities_collection,
            "community_id_property": community_id_property,
            "summary_property": summary_property,
            "size_property": size_property,
            "level_property": level_property,
        }
        if limit is not None:
            bind.update({"skip": skip, "limit": limit})
        aql = f"""
        FOR c IN @@communities
          {filter_clause}
          SORT c[@community_id_property] ASC
          {limit_clause}
          RETURN {{
              id: c[@community_id_property],
              summary: c[@summary_property],
              size: c[@size_property],
              level: c[@level_property],
              document: c
          }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    @staticmethod
    def get_unique_table_headers(
        db,
        tables_collection: str,
        headers_property: str,
    ) -> List[List[str]]:
        aql = f"""
        FOR t IN @@tables
          FILTER HAS(t, @hp)
          COLLECT h = t[@hp]
          RETURN h
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars={"hp": headers_property, "@tables": tables_collection})))

    # --------------------------
    # Bridge / GNN Integration Methods (Mirrored from GlobalGraphAccessor)
    # --------------------------

    def _signal_query(self, query, bind):
        from odin.backends.base import BackendIOError
        try:
            return list(self.db.aql.execute(query, bind_vars=bind))
        except Exception as exc:
            raise BackendIOError("Could not read configured Arango community signal") from exc

    def _algorithm_filter(self, alias, field, bind):
        if field is None:
            return ""
        bind["algorithm_field"] = field
        bind["algorithm"] = self.algorithm
        return f"FILTER {alias}[@algorithm_field] == @algorithm"

    def is_bridge(self, entity_id: str) -> Optional[dict]:
        """Read a bridge by its full document ID; errors are never cached."""
        if self.bridge_col is None:
            return None
        if entity_id in self._bridge_cache:
            return self._bridge_cache[entity_id]
        bind = {"@bridge_col": self.bridge_col, "entity_id": entity_id,
                "entity_field": self.bridge_entity_field}
        guard = self._algorithm_filter("b", self.bridge_algorithm_field, bind)
        result = self._signal_query(f"""
        FOR b IN @@bridge_col
          FILTER b[@entity_field] == @entity_id
          {guard}
          RETURN b
        """, bind)
        bridge = result[0] if result else None
        if bridge is not None:
            # Keep the complete record and expose the configured numeric signal.
            bridge = {"record": bridge, "bridge_strength": bridge[self.bridge_strength_field]}
        self._bridge_cache[entity_id] = bridge
        return bridge

    def get_entity_community(self, entity_id: str) -> Optional[str]:
        """Membership lookup is independent of traversal scope."""
        if self.membership_col:
            bind = {"@membership_col": self.membership_col, "entity_id": entity_id,
                    "membership_entity_field": self.memb_ent_field,
                    "membership_community_field": self.memb_com_field}
            guard = self._algorithm_filter("m", self.membership_algorithm_field, bind)
            result = self._signal_query(f"""
            FOR m IN @@membership_col
              FILTER m[@membership_entity_field] == @entity_id
              {guard}
              RETURN m[@membership_community_field]
            """, bind)
            return result[0] if result else None
        if self.community_prop:
            result = self._signal_query(
                "LET d = DOCUMENT(@id) RETURN d[@community_field]",
                {"id": entity_id, "community_field": self.community_prop})
            return result[0] if result else None
        return None

    def get_affinity(self, community_a: str, community_b: str) -> float:
        """Read an explicitly mapped affinity; missing rows have zero signal."""
        if self.affinity_col is None or not community_a or not community_b:
            return 0.0
        cache_key = tuple(sorted((community_a, community_b)))
        if cache_key in self._affinity_cache:
            return self._affinity_cache[cache_key]
        bind = {"@affinity_col": self.affinity_col, "comm_a": community_a,
                "comm_b": community_b, "from_field": self.affinity_from_field,
                "to_field": self.affinity_to_field, "score_field": self.affinity_score_field}
        guard = self._algorithm_filter("a", self.affinity_algorithm_field, bind)
        result = self._signal_query(f"""
        FOR a IN @@affinity_col
          {guard}
          FILTER (a[@from_field] == @comm_a AND a[@to_field] == @comm_b)
              OR (a[@from_field] == @comm_b AND a[@to_field] == @comm_a)
          RETURN a[@score_field]
        """, bind)
        affinity = float(result[0]) if result else 0.0
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
        membership_collection: str,
        membership_entity_field: str,
        membership_community_field: str,
        edges_collection: str,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Get top entities by degree WITHIN a specific community.
        Essential for autonomous discovery - provides high-value seed nodes.
        
        Returns:
            List of {entity: str, degree: int}
        """
        limit_clause = "LIMIT @limit" if limit is not None else ""
        bind: Dict[str, Any] = {
            "@membership": membership_collection,
            "@edges": edges_collection,
            "m_ent": membership_entity_field,
            "m_com": membership_community_field,
            "cid": community_id,
        }
        if limit is not None:
            bind["limit"] = limit
        aql = f"""
        LET community_entities = (
            FOR m IN @@membership
                FILTER m[@m_com] == @cid
                RETURN m[@m_ent]
        )
        FOR e IN @@edges
            FILTER e._from IN community_entities
            COLLECT entity = e._from WITH COUNT INTO degree
            SORT degree DESC
            {limit_clause}
            RETURN {{ entity: entity, degree: degree }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    @staticmethod
    def get_recent_entities(
        db,
        since: str,  # ISO timestamp
        community_id: Optional[str] = None,
        *,
        nodes_collection: str,
        membership_collection: Optional[str] = None,
        membership_entity_field: Optional[str] = None,
        membership_community_field: Optional[str] = None,
        created_at_property: str,
        updated_at_property: str,
        limit: Optional[int] = None,
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
            "@nodes": nodes_collection,
            "created_prop": created_at_property,
            "updated_prop": updated_at_property,
        }
        
        community_filter = ""
        if community_id:
            if not all((membership_collection, membership_entity_field,
                        membership_community_field)):
                raise ValueError("community filtering requires membership mapping fields")
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

        limit_clause = "LIMIT @limit" if limit is not None else ""
        if limit is not None:
            bind["limit"] = limit
        
        aql = f"""
        FOR e IN @@nodes
            FILTER (HAS(e, @created_prop) AND e[@created_prop] >= @since)
                OR (HAS(e, @updated_prop) AND e[@updated_prop] >= @since)
            {community_filter}
            SORT HAS(e, @created_prop) ? e[@created_prop] : e[@updated_prop] DESC
            {limit_clause}
            RETURN {{
                entity: e._id,
                created_at: HAS(e, @created_prop) ? e[@created_prop] : null,
                updated_at: HAS(e, @updated_prop) ? e[@updated_prop] : null,
                document: e
            }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    @staticmethod
    def search_entities(
        db,
        query: str,
        community_id: Optional[str] = None,
        *,
        nodes_collection: str,
        membership_collection: Optional[str] = None,
        membership_entity_field: Optional[str] = None,
        membership_community_field: Optional[str] = None,
        search_fields: List[str],
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Text search for entities matching query.
        Uses LIKE for simple text matching (can be upgraded to ArangoSearch).
        
        Args:
            query: Search string
            search_fields: Fields to search.
            
        Returns:
            List of complete matching entity documents and matched field names.
        """
        if not search_fields:
            raise ValueError("search_fields must contain at least one field")
        bind: Dict[str, Any] = {
            "query": f"%{query.lower()}%",
            "search_fields": search_fields,
            "@nodes": nodes_collection,
        }
        
        community_filter = ""
        if community_id:
            if not all((membership_collection, membership_entity_field,
                        membership_community_field)):
                raise ValueError("community filtering requires membership mapping fields")
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

        limit_clause = "LIMIT @limit" if limit is not None else ""
        if limit is not None:
            bind["limit"] = limit
        
        aql = f"""
        FOR e IN @@nodes
            LET matched_fields = (
                FOR field IN @search_fields
                    FILTER HAS(e, field) AND LOWER(TO_STRING(e[field])) LIKE @query
                    RETURN field
            )
            FILTER LENGTH(matched_fields) > 0
            {community_filter}
            {limit_clause}
            RETURN {{
                entity: e._id,
                matched_fields: matched_fields,
                document: e
            }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars=bind)))

    # ════════════════════════════════════════════════════════════════
    # CONTENT HYDRATION (for agent reasoning)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def get_document_content(
        db,
        doc_id: str,
        *,
        text_collection: str,
        table_collection: str,
        image_collection: str,
        document_collection: str,
    ) -> Optional[dict]:
        """
        Fetch content from any document collection by ID.
        Essential for agent reasoning - converts graph IDs to actual content.
        
        Args:
            doc_id: Document ID in format "CollectionName/key"
            
        Returns:
            The complete document and its configured collection, or None if not found.
        """
        try:
            collection, key = doc_id.split("/", 1)
        except ValueError:
            return None
        
        if collection not in {
            text_collection,
            table_collection,
            image_collection,
            document_collection,
        }:
            return None

        aql = """
        FOR source IN @@collection
            FILTER source._id == @doc_id
            RETURN {
                source_id: source._id,
                source_type: @source_type,
                document: source
            }
        """
        result = list(db.aql.execute(aql, bind_vars={
            "@collection": collection,
            "doc_id": doc_id,
            "source_type": collection,
        }))
        return clean_evidence(result[0]) if result else None

    @staticmethod
    def get_entity_sources(
        db,
        entity_id: str,
        *,
        extracted_from_collection: str,
    ) -> List[dict]:
        """
        Get all sources for an entity through the configured provenance edges.
        Critical for evidence gathering - shows WHERE an entity was mentioned.
        
        Args:
            entity_id: Arango document ID (e.g., "CaseRecords/ent_123")
        Returns:
            Complete provenance-edge and source-document records.
        """
        aql = f"""
        FOR edge IN @@provenance_edges
            FILTER edge._from == @entity_id
            LET source = DOCUMENT(edge._to)
            FILTER source != null
            LET collection = PARSE_IDENTIFIER(edge._to).collection
            RETURN {{
                source_id: edge._to,
                source_type: collection,
                edge: edge,
                document: source
            }}
        """
        return clean_evidence(list(db.aql.execute(aql, bind_vars={
            "entity_id": entity_id, "@provenance_edges": extracted_from_collection,
        })))

    @staticmethod
    def search_content(
        db,
        query: str,
        *,
        content_types: Optional[List[str]] = None,
        text_collection: str,
        table_collection: str,
        image_collection: str,
        text_search_fields: List[str],
        table_search_fields: List[str],
        image_search_fields: List[str],
    ) -> List[dict]:
        """
        Semantic/text search across content collections.
        Uses simple LIKE matching (can be upgraded to ArangoSearch/vectors).
        
        Args:
            query: Search string
            content_types: Collections to search; defaults to every explicitly
                supplied content collection.
            *_search_fields: Fields searched in the corresponding collection.
            
        Returns:
            Complete matching source documents.
        """
        if content_types is None:
            content_types = [text_collection, table_collection, image_collection]
        requested_collections = {
            text_collection: text_search_fields,
            table_collection: table_search_fields,
            image_collection: image_search_fields,
        }
        for collection in content_types:
            if collection not in requested_collections:
                raise ValueError(f"unknown content collection: {collection}")
            if not requested_collections[collection]:
                raise ValueError(f"search fields are required for {collection}")

        results = []
        for collection in content_types:
            query_aql = """
            FOR document IN @@sources
              LET matched = (
                FOR field IN @search_fields
                  FILTER HAS(document, field) AND LOWER(TO_STRING(document[field])) LIKE @query
                  RETURN field
              )
              FILTER LENGTH(matched) > 0
              RETURN {source_id: document._id, source_type: @source_type, document: document}
            """
            results.extend(list(db.aql.execute(query_aql, bind_vars={
                "@sources": collection, "source_type": collection,
                "search_fields": requested_collections[collection],
                "query": f"%{query.lower()}%",
            })))

        return clean_evidence(results)

    # --------------------------
    # Internal neighbor routine
    # --------------------------
    def _iter_neighbors(self, node: NodeId, *, direction: str, rich: bool) -> Iterable[EdgeView]:
        assert direction in ("OUTBOUND", "INBOUND")

        bind: Dict[str, Any] = {
            "node": node,
            "@edges": self.edges_col,
            "rel_prop": self.rel_prop,
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
            bind["community_field"] = self.community_prop
            filters.append("v[@community_field] == @cid")
        elif self.community_mode == "mapping":
            bind.update({"@mcol": self.membership_col, "m_ent": self.memb_ent_field, "m_com": self.memb_com_field})
            guard = self._algorithm_filter("m", self.membership_algorithm_field, bind)
            filters.append(f"""
              FIRST(
                FOR m IN @@mcol
                  FILTER m[@m_com] == @cid AND m[@m_ent] == v._id
                  {guard}
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
            bind["node_type_field"] = self.node_type_prop
            filters.append("v[@node_type_field] IN @allowed_neighbor_types")

        # Time window filter on edge timestamp
        if self.time_window and self.ts_prop:
            bind["start_ts"], bind["end_ts"] = self.time_window
            bind["ts_prop"] = self.ts_prop
            filters.append("HAS(e, @ts_prop) AND e[@ts_prop] >= @start_ts AND e[@ts_prop] <= @end_ts")

        # Validity fields are strictly opt-in; an active filter needs a mapping.
        if self.current_only:
            if not self.as_of or not (self.edge_valid_from_prop or self.edge_valid_to_prop):
                raise ValueError("current_only requires as_of and a configured validity field")
            bind["as_of"] = self.as_of
            if self.edge_valid_from_prop:
                bind["valid_from_field"] = self.edge_valid_from_prop
                filters.append("(e[@valid_from_field] == null OR e[@valid_from_field] <= @as_of)")
            if self.edge_valid_to_prop:
                bind["valid_to_field"] = self.edge_valid_to_prop
                filters.append("(e[@valid_to_field] == null OR e[@valid_to_field] >= @as_of)")

        # Mapping status exposes metadata. Filtering requires explicit allowed values.
        if self.allowed_edge_statuses is not None:
            if not self.edge_status_prop:
                raise ValueError("allowed_edge_statuses requires edge_status_property")
            bind["status_field"] = self.edge_status_prop
            bind["allowed_statuses"] = self.allowed_edge_statuses
            filters.append("e[@status_field] IN @allowed_statuses")

        # Recency decay: 2^(- age_days / half_life)
        recency_clause = "1.0"
        if self.recency_half_life_days is not None and self.as_of and self.ts_prop:
            bind["half_life"] = float(self.recency_half_life_days)
            bind["as_of"] = self.as_of
            bind["ts_prop"] = self.ts_prop
            recency_clause = "POW(2, -1 * DATE_DIFF(@as_of, e[@ts_prop], 'days') / @half_life)"

        # Base weight
        weight_clause = "1.0"
        if self.w_prop is not None:
            bind["w_prop"] = self.w_prop
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
            FOR p IN @@provenance_edges
              FILTER p._from IN [e._from, e._to]
              FILTER LENGTH(@provenance_targets) == 0 OR PARSE_IDENTIFIER(p._to).collection IN @provenance_targets
              RETURN p._to
            """
            if self.prov_edges_col else "[]"
        )

        if self.prov_edges_col:
            bind["@provenance_edges"] = self.prov_edges_col
            bind["provenance_targets"] = self.prov_target_cols
        expressions = {}
        for name, field in (("vf", self.edge_valid_from_prop), ("vt", self.edge_valid_to_prop),
                            ("status", self.edge_status_prop), ("timestamp", self.ts_prop),
                            ("raw_confidence", self.edge_raw_conf_prop),
                            ("npll_posterior", self.edge_npll_post_prop),
                            ("calibration", self.edge_calibration_prop)):
            expressions[name] = "null"
            if field:
                bind[f"meta_{name}"] = field
                expressions[name] = f"e[@meta_{name}]"
        bind["inline_provenance_fields"] = self.edge_prov_fields

        aql = f"""
        LET priors = @priors_map
        FOR v, e IN 1..1 {direction} @node @@edges
          {hint}
          FILTER {filters_str}
          LET _rel = e[@rel_prop]
          LET _prior = TO_NUMBER(NOT_NULL(priors[_rel], 1.0))
          LET _base_w = {weight_clause}
          LET _rec    = {recency_clause}
          LET _conf   = {conf_clause}
          LET _w_eff  = TO_NUMBER(_base_w) * TO_NUMBER(_prior) * TO_NUMBER(_rec) * TO_NUMBER(_conf)

          LET _vf = {expressions["vf"]}
          LET _vt = {expressions["vt"]}
          LET _status2 = {expressions["status"]}

          // Provenance: configured inline fields and provenance edges for both endpoints
          LET _src_inline_candidates = (FOR field IN @inline_provenance_fields RETURN e[field])
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
            timestamp: {expressions["timestamp"]},
            valid_from: _vf,
            valid_to: _vt,
            status: _status2,
            raw_confidence: {expressions["raw_confidence"]},
            npll_posterior: {expressions["npll_posterior"]},
            calibration: {expressions["calibration"]},
            sources: _sources,
            assertion: e
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
                    assertion=d["assertion"],
                    timestamp=d.get("timestamp"),
                )
            else:
                yield d["v_id"], d["rel"], float(d["weight"])


class GlobalGraphAccessor(ArangoCommunityAccessor):
    """Unscoped graph access with the same explicit mapping and evidence contract."""

    def __init__(self, db, **mapping):
        super().__init__(db, community_id="global", community_mode="none", **mapping)

    def get_bridges_from_community(self, community_id: str, min_strength: int = 1):
        bind = {"@bridge_col": self.bridge_col, "community_field": self.bridge_community_field,
                "strength_field": self.bridge_strength_field, "community_id": community_id,
                "min_strength": min_strength}
        guard = self._algorithm_filter("b", self.bridge_algorithm_field, bind)
        return self._signal_query(f"""
        FOR b IN @@bridge_col
          {guard}
          FILTER b[@community_field] == @community_id
          FILTER b[@strength_field] >= @min_strength
          SORT b[@strength_field] DESC
          RETURN b
        """, bind)

    def get_top_bridges(self, limit: int = 20):
        bind = {"@bridge_col": self.bridge_col, "strength_field": self.bridge_strength_field,
                "limit": limit}
        guard = self._algorithm_filter("b", self.bridge_algorithm_field, bind)
        return self._signal_query(f"""
        FOR b IN @@bridge_col
          {guard}
          SORT b[@strength_field] DESC
          LIMIT @limit
          RETURN b
        """, bind)

    def get_strongest_affinities(self, limit: int = 20):
        bind = {"@affinity_col": self.affinity_col, "score_field": self.affinity_score_field,
                "limit": limit}
        guard = self._algorithm_filter("a", self.affinity_algorithm_field, bind)
        return self._signal_query(f"""
        FOR a IN @@affinity_col
          {guard}
          SORT a[@score_field] DESC
          LIMIT @limit
          RETURN a
        """, bind)
