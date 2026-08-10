import re
from typing import Any, Dict

# Simple regex patterns for common PII
PII_PATTERNS = {
    'EMAIL': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'PHONE': re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
}

def redact_text(text: str) -> str:
    """Redacts PII from a single string."""
    if not isinstance(text, str):
        return text
    for pii_type, pattern in PII_PATTERNS.items():
        text = pattern.sub(f'[{pii_type}_REDACTED]', text)
    return text

def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redacts PII from string values in a dictionary."""
    if not isinstance(data, dict):
        return data
    
    clean_dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            clean_dict[key] = redact_text(value)
        elif isinstance(value, dict):
            clean_dict[key] = redact_dict(value)
        elif isinstance(value, list):
            clean_dict[key] = [
                redact_dict(item) if isinstance(item, dict) else (redact_text(item) if isinstance(item, str) else item)
                for item in value
            ]
        else:
            clean_dict[key] = value
    return clean_dict
