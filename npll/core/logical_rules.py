"""
Logical Rules and Ground Rules for NPLL
"""

from typing import List, Set, Dict, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum
import itertools
import logging
from collections import defaultdict

from .knowledge_graph import Entity, Relation, Triple, KnowledgeGraph

logger = logging.getLogger(__name__)


class Variable:
    """Variable in logical rules"""
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Var({self.name})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class Atom:
    """
    Atomic formula in first-order logic: Pred(arg1, arg2)
    """
    predicate: Relation
    arguments: Tuple[Any, ...]  
    
    def __post_init__(self):
        # Validate arguments
        for arg in self.arguments:
            if not isinstance(arg, (Variable, Entity)):
                raise ValueError(f"Atom arguments must be Variable or Entity, got {type(arg)}")
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.predicate.name}({args_str})"
    
    def is_ground(self) -> bool:
        """Check if atom is ground (no variables)"""
        return all(isinstance(arg, Entity) for arg in self.arguments)
    
    def get_variables(self) -> Set[Variable]:
        """Get all variables in this atom"""
        return {arg for arg in self.arguments if isinstance(arg, Variable)}
    
    def ground_with_substitution(self, substitution: Dict[Variable, Entity]) -> 'Atom':
        """Ground atom by substituting variables with entities"""
        new_args = []
        for arg in self.arguments:
            if isinstance(arg, Variable) and arg in substitution:
                new_args.append(substitution[arg])
            else:
                new_args.append(arg)
        
        return Atom(predicate=self.predicate, arguments=tuple(new_args))
    
    def to_triple(self) -> Optional[Triple]:
        """Convert ground atom to triple (if binary and ground)"""
        if len(self.arguments) == 2 and self.is_ground():
            head, tail = self.arguments
            return Triple(head=head, relation=self.predicate, tail=tail)
        return None


class RuleType(Enum):
    """Types of logical rules"""
    HORN_CLAUSE = "horn"  # Standard Horn clause
    EQUALITY = "equality"  # Equality rules
    TRANSITIVITY = "transitivity"  # Transitive rules
    SYMMETRY = "symmetry"  # Symmetric rules
    PRIOR = "prior"  # Unconditional relation prior
    GENERAL = "general"  # General first-order rules


@dataclass
class LogicalRule:
    """
    First-order logical rule: Body ⇒ Head
    """
    
    rule_id: str
    body: List[Atom]  # Body atoms (premise)
    head: Atom  # Head atom (conclusion)
    rule_type: RuleType = RuleType.GENERAL
    confidence: float = 0.5  # Initial confidence score
    support: int = 0  # Number of supporting instances
    
    def __post_init__(self):
        """Validate rule structure"""
        if not self.body and self.rule_type != RuleType.PRIOR:
            raise ValueError("Only relation prior rules may have an empty body")
        
        if not isinstance(self.head, Atom):
            raise ValueError("Rule head must be an Atom")
        
        # Check variable consistency
        body_vars = set()
        for atom in self.body:
            body_vars.update(atom.get_variables())
        
        head_vars = self.head.get_variables()
        
        # All head variables should appear in body
        if self.rule_type != RuleType.PRIOR and not head_vars.issubset(body_vars):
            logger.warning(f"Rule {self.rule_id}: Head variables not in body")
    
    def get_all_variables(self) -> Set[Variable]:
        """Get all variables in the rule"""
        variables = set()
        for atom in self.body:
            variables.update(atom.get_variables())
        variables.update(self.head.get_variables())
        return variables
    
    def get_predicates(self) -> Set[Relation]:
        """Get all predicates used in the rule"""
        predicates = {atom.predicate for atom in self.body}
        predicates.add(self.head.predicate)
        return predicates
    
    def generate_ground_rules(self, kg: KnowledgeGraph, 
                            max_groundings: int = 1000) -> List['GroundRule']:
        """
        Generate ground rules by substituting variables with entities
        """
        ground_rules = []
        entities = sorted(kg.entities, key=lambda entity: entity.name)
        
        if not entities:
            return ground_rules
        
        # Get all variables that need to be substituted
        variables = sorted(self.get_all_variables(), key=lambda variable: variable.name)
        
        if not variables:
            # Already ground rule
            ground_body = [atom.to_triple() for atom in self.body if atom.is_ground()]
            ground_head_triple = self.head.to_triple()
            
            if ground_head_triple and all(t is not None for t in ground_body):
                ground_rules.append(GroundRule(
                    rule_id=self.rule_id,
                    body_facts=[t for t in ground_body if t is not None],
                    head_fact=ground_head_triple,
                    parent_rule=self
                ))
            return ground_rules
        
        # Generate all possible variable substitutions
        # Limit combinations to prevent explosion
        max_entities_per_var = min(len(entities), max_groundings // len(variables) + 1)
        
        substitution_count = 0
        for substitution_values in itertools.product(entities[:max_entities_per_var], 
                                                   repeat=len(variables)):
            if substitution_count >= max_groundings:
                break
            
            substitution = dict(zip(variables, substitution_values))
            
            # Generate ground atoms
            try:
                ground_body_atoms = [atom.ground_with_substitution(substitution) 
                                   for atom in self.body]
                ground_head_atom = self.head.ground_with_substitution(substitution)
                
                # Convert to triples
                ground_body_triples = [atom.to_triple() for atom in ground_body_atoms 
                                     if atom.is_ground()]
                ground_head_triple = ground_head_atom.to_triple()
                
                # Check if all conversions successful
                if (ground_head_triple and 
                    len(ground_body_triples) == len(ground_body_atoms) and
                    all(t is not None for t in ground_body_triples)):
                    
                    ground_rule = GroundRule(
                        rule_id=f"{self.rule_id}_ground_{substitution_count}",
                        body_facts=ground_body_triples,
                        head_fact=ground_head_triple,
                        parent_rule=self,
                        substitution=substitution.copy()
                    )
                    
                    ground_rules.append(ground_rule)
                    substitution_count += 1
                    
            except Exception as e:
                logger.debug(f"Failed to ground rule {self.rule_id}: {e}")
                continue
        
        logger.info(f"Generated {len(ground_rules)} ground rules for {self.rule_id}")
        return ground_rules
    
    def to_cnf(self) -> str:
        """
        Convert rule to Conjunctive Normal Form (CNF)
        """
        # Body atoms become negated disjuncts
        cnf_parts = [f"¬{atom}" for atom in self.body]
        # Head atom becomes positive disjunct
        cnf_parts.append(str(self.head))
        
        return " ∨ ".join(cnf_parts)
    
    def __str__(self) -> str:
        body_str = " ∧ ".join(str(atom) for atom in self.body)
        return f"{body_str} ⇒ {self.head}"
    
    def __repr__(self) -> str:
        return f"LogicalRule(id={self.rule_id}, {self})"

    # --- Serialization helpers ---
    def serialize(self) -> Dict[str, Any]:
        def _atom_to_dict(atom: Atom) -> Dict[str, Any]:
            args = []
            for a in atom.arguments:
                if isinstance(a, Variable):
                    args.append({'type': 'var', 'name': a.name})
                elif isinstance(a, Entity):
                    args.append({'type': 'ent', 'name': a.name})
                else:
                    args.append({'type': 'raw', 'value': str(a)})
            return {'predicate': atom.predicate.name, 'arguments': args}

        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type.value,
            'confidence': self.confidence,
            'support': self.support,
            'body': [_atom_to_dict(a) for a in self.body],
            'head': _atom_to_dict(self.head),
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'LogicalRule':
        rule_id = data['rule_id']
        rule_type = RuleType(data.get('rule_type', RuleType.GENERAL.value))
        confidence = data.get('confidence', 0.5)
        support = data.get('support', 0)

        def _atom_from_dict(d: Dict[str, Any]) -> Atom:
            pred = Relation(d['predicate'])
            args = []
            for a in d['arguments']:
                if a.get('type') == 'var':
                    args.append(Variable(a['name']))
                elif a.get('type') == 'ent':
                    args.append(Entity(a['name']))
                else:
                    # Fallback: treat as entity by name
                    args.append(Entity(str(a.get('value', ''))))
            return Atom(predicate=pred, arguments=tuple(args))

        body_atoms = [_atom_from_dict(x) for x in data['body']]
        head_atom = _atom_from_dict(data['head'])
        return LogicalRule(
            rule_id=rule_id,
            body=body_atoms,
            head=head_atom,
            rule_type=rule_type,
            confidence=confidence,
            support=support,
        )


@dataclass
class GroundRule:
    """
    Ground instance of a logical rule with all variables substituted
    
    """
    
    rule_id: str
    body_facts: List[Triple]  # Ground body facts
    head_fact: Triple  # Ground head fact
    parent_rule: LogicalRule  # Original rule this was grounded from
    substitution: Optional[Dict[Variable, Entity]] = None
    
    def __post_init__(self):
        """Validate ground rule"""
        if not self.body_facts and self.parent_rule.rule_type != RuleType.PRIOR:
            raise ValueError("Only relation prior ground rules may have an empty body")
        
        if not isinstance(self.head_fact, Triple):
            raise ValueError("Ground rule head must be a Triple")
    
    def evaluate_truth_value(self, kg: KnowledgeGraph) -> bool:
        """
        Evaluate truth value of ground rule given knowledge graph
        """
        # Check if all body facts are true in KG
        body_satisfied = all(
            kg.contains_fact(fact.head.name, fact.relation.name, fact.tail.name)
            for fact in self.body_facts
        )
        
        # If body is false, rule is vacuously true
        if not body_satisfied:
            return True
        
        # If body is true, check if head is true
        head_satisfied = kg.contains_fact(
            self.head_fact.head.name, 
            self.head_fact.relation.name, 
            self.head_fact.tail.name
        )
        
        return head_satisfied
    
    def get_all_facts(self) -> List[Triple]:
        """Get all facts (body + head) in this ground rule"""
        return self.body_facts + [self.head_fact]
    
    def get_fact_truth_values(self, kg: KnowledgeGraph) -> Dict[Triple, bool]:
        """Get truth values for all facts in this ground rule"""
        truth_values = {}
        
        for fact in self.body_facts:
            truth_values[fact] = kg.contains_fact(
                fact.head.name, fact.relation.name, fact.tail.name
            )
        
        truth_values[self.head_fact] = kg.contains_fact(
            self.head_fact.head.name, 
            self.head_fact.relation.name, 
            self.head_fact.tail.name
        )
        
        return truth_values
    
    def __str__(self) -> str:
        body_str = " ∧ ".join(str(fact) for fact in self.body_facts)
        return f"{body_str} ⇒ {self.head_fact}"
    
    def __repr__(self) -> str:
        return f"GroundRule(id={self.rule_id}, {self})"


class RuleGenerator:
    """Mine rules from observed facts, never from vacuously true implications."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.facts = {(f.head.name, f.relation.name, f.tail.name)
                      for f in kg.known_facts | kg.unknown_facts}
        self.relations = {r.name: r for r in kg.relations}

    @staticmethod
    def _id(kind, names):
        import hashlib
        import json
        return kind + "_" + hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest()

    def generate_relation_priors(self):
        """Every relation gets a non-tautological prior, independent of mined motifs."""
        x, y = Variable("x"), Variable("y")
        possible = max(1, len(self.kg.entities) ** 2)
        rules = []
        for name, relation in sorted(self.relations.items()):
            support = sum(1 for _, r, _ in self.facts if r == name)
            rules.append(LogicalRule(
                self._id("prior", [name]), [], Atom(relation, (x, y)),
                rule_type=RuleType.PRIOR, confidence=support / possible, support=support))
        return rules

    def generate_simple_rules(self, min_support=2, max_rule_length=3):
        """Count complete observed two-edge joins and their observed conclusions."""
        from collections import defaultdict
        outgoing = defaultdict(list)
        conclusions = defaultdict(set)
        for u, r, v in sorted(self.facts):
            outgoing[u].append((r, v))
            conclusions[(u, v)].add(r)
        bodies = defaultdict(set)
        supports = defaultdict(set)
        for u, r1, v in sorted(self.facts):
            for r2, w in outgoing[v]:
                bodies[(r1, r2)].add((u, v, w))
                for r3 in conclusions[(u, w)]:
                    supports[(r1, r2, r3)].add((u, v, w))
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        rules = []
        for names, instances in sorted(supports.items()):
            if len(instances) < min_support:
                continue
            r1, r2, r3 = (self.relations[name] for name in names)
            rules.append(LogicalRule(
                self._id("chain", names), [Atom(r1, (x, y)), Atom(r2, (y, z))],
                Atom(r3, (x, z)), rule_type=RuleType.TRANSITIVITY,
                confidence=len(instances) / len(bodies[names[:2]]), support=len(instances)))
        return rules

    def generate_symmetry_rules(self, min_support=2):
        x, y = Variable("x"), Variable("y")
        rules = []
        for name, relation in sorted(self.relations.items()):
            pairs = {(u, v) for u, r, v in self.facts if r == name and u != v}
            support = sum(1 for u, v in pairs if (v, u) in pairs)
            if support >= min_support:
                rules.append(LogicalRule(
                    self._id("symmetry", [name]), [Atom(relation, (x, y))],
                    Atom(relation, (y, x)), rule_type=RuleType.SYMMETRY,
                    confidence=support / len(pairs), support=support))
        return rules


def parse_rule_from_string(rule_str: str, entities: Dict[str, Entity], 
                          relations: Dict[str, Relation]) -> LogicalRule:
    """
    Parse logical rule from string format
    """
    # Split by implication arrow
    if "⇒" not in rule_str:
        raise ValueError(f"Rule must contain ⇒: {rule_str}")
    
    body_str, head_str = rule_str.split("⇒", 1)
    
    # Parse body atoms (split by ∧)
    body_atom_strs = [atom.strip() for atom in body_str.split("∧")]
    body_atoms = []
    
    for atom_str in body_atom_strs:
        atom = _parse_atom_from_string(atom_str.strip(), entities, relations)
        body_atoms.append(atom)
    
    # Parse head atom
    head_atom = _parse_atom_from_string(head_str.strip(), entities, relations)
    
    rule_id = f"parsed_{hash(rule_str) % 10000}"
    
    return LogicalRule(
        rule_id=rule_id,
        body=body_atoms,
        head=head_atom,
        rule_type=RuleType.GENERAL,
        confidence=0.5
    )


def _parse_atom_from_string(atom_str: str, entities: Dict[str, Entity], 
                           relations: Dict[str, Relation]) -> Atom:
    """Parse single atom from string: 'Pred(arg1,arg2)'"""
    # Extract predicate and arguments
    if "(" not in atom_str or ")" not in atom_str:
        raise ValueError(f"Invalid atom format: {atom_str}")
    
    pred_name = atom_str[:atom_str.index("(")].strip()
    args_str = atom_str[atom_str.index("(")+1:atom_str.rindex(")")].strip()
    
    # Get or create relation
    if pred_name not in relations:
        relations[pred_name] = Relation(name=pred_name)
    predicate = relations[pred_name]
    
    # Parse arguments
    arg_names = [arg.strip() for arg in args_str.split(",")]
    arguments = []
    
    for arg_name in arg_names:
        # Check if it's a variable (lowercase) or entity
        if arg_name.islower() or arg_name.startswith('?'):
            arguments.append(Variable(arg_name))
        else:
            # Entity - create if doesn't exist
            if arg_name not in entities:
                entities[arg_name] = Entity(name=arg_name)
            arguments.append(entities[arg_name])
    
    return Atom(predicate=predicate, arguments=tuple(arguments))