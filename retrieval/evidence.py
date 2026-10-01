"""Preserve complete reasoning evidence while excluding internal vector fields."""

VECTOR_FIELDS = frozenset({"embedding", "npll_embedding", "transe_embedding"})


def exclude_vectors(value, path=""):
    excluded = []
    def visit(item, current):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                child_path = f"{current}.{key}" if current else key
                if key in VECTOR_FIELDS:
                    excluded.append(child_path)
                else:
                    result[key] = visit(child, child_path)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, f"{current}[{index}]") for index, child in enumerate(item)]
        return item
    return visit(value, path), excluded
