from __future__ import annotations

import json
import os
from typing import Any


class LLMAdapter:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def json_response(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self._client:
            return None

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return None

    def text_response(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self._client:
            return None

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception:
            return None
