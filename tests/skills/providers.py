"""Model providers for skill evals. Pluggable; no third-party dependencies."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request


class AnthropicProvider:
    """Anthropic Messages API over HTTPS. Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
        return "".join(part.get("text", "") for part in payload["content"])


class CommandProvider:
    """Any CLI that reads a prompt on stdin and prints the answer, e.g. `claude -p`."""

    def __init__(self, command: str):
        self.command = command

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n---\n\n{user}" if system else user
        done = subprocess.run(self.command, shell=True, input=prompt, capture_output=True,
                              text=True, timeout=600, check=True)
        return done.stdout


def make_provider(kind: str, model: str = "", command: str = ""):
    if kind == "anthropic":
        return AnthropicProvider(model)
    if kind == "command":
        return CommandProvider(command)
    raise ValueError(f"unknown provider {kind!r}")
