"""Model providers for skill evals. SDKs are imported lazily (optional `evals` extra)."""
from __future__ import annotations

import os
import subprocess

TIMEOUT = 120.0  # seconds per request; a stuck call fails instead of hanging the run


class AnthropicProvider:
    """Anthropic Messages API. Reads ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_tokens: int = 4096):
        import anthropic

        self.client = anthropic.Anthropic(timeout=TIMEOUT, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        kwargs = {"system": system} if system else {}
        reply = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user}], **kwargs)
        return "".join(b.text for b in reply.content if b.type == "text")


class _ChatProvider:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def complete(self, system: str, user: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": user}]
        reply = self.client.chat.completions.create(model=self.model, messages=messages)
        return reply.choices[0].message.content or ""


class OpenAIProvider(_ChatProvider):
    """OpenAI chat completions. Reads OPENAI_API_KEY."""

    def __init__(self, model: str):
        import openai

        super().__init__(openai.OpenAI(timeout=TIMEOUT, max_retries=2), model)


class AzureOpenAIProvider(_ChatProvider):
    """Azure OpenAI. Reads AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION and, unless a
    model is given, AZURE_DEPLOYMENT (the deployment name is the model)."""

    def __init__(self, deployment: str = ""):
        import openai

        client = openai.AzureOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["AZURE_API_KEY"],
            api_version=os.environ["AZURE_API_VERSION"],
            timeout=TIMEOUT,
            max_retries=2,
        )
        super().__init__(client, deployment or os.environ["AZURE_DEPLOYMENT"])


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
    if kind == "openai":
        return OpenAIProvider(model)
    if kind == "azure":
        return AzureOpenAIProvider(model)
    if kind == "command":
        return CommandProvider(command)
    raise ValueError(f"unknown provider {kind!r}")
