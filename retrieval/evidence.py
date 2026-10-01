"""Preserve complete reasoning evidence while excluding internal vector fields."""

VECTOR_FIELDS = frozenset({"embedding", "npll_embedding", "transe_embedding"})


def exclude_vectors(value, path=""):
    excluded = []
    def numeric_vector(item):
        return isinstance(item, (list, tuple)) and all(
            type(value) in (int, float) or numeric_vector(value) for value in item)

    def visit(item, current):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                child_path = f"{current}.{key}" if current else key
                if key in VECTOR_FIELDS and numeric_vector(child):
                    excluded.append(child_path)
                else:
                    result[key] = visit(child, child_path)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, f"{current}[{index}]") for index, child in enumerate(item)]
        return item
    return visit(value, path), excluded


def clean_evidence(value):
    """Remove only exempt vectors and declare every excluded path on each record."""
    if isinstance(value, list):
        return [clean_evidence(record) for record in value]
    clean, excluded = exclude_vectors(value)
    if excluded and isinstance(clean, dict):
        if "odin_excluded_vector_fields" in clean:
            raise ValueError("Evidence contains the reserved vector-exclusion metadata field")
        clean["odin_excluded_vector_fields"] = excluded
    return clean
