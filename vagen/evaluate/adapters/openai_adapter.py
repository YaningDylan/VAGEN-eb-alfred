# All comments are in English.
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple
from PIL import Image
from vagen.evaluate.adapters.base_adapter import ModelAdapter
from vagen.evaluate.utils.mm_utils import pil_to_dataurl_png, compile_text_images_for_order

class OpenAIAdapter(ModelAdapter):
    """
    OpenAI-compatible multimodal adapter:
    - messages use content parts with {"type": "text"} and {"type": "image_url"}.
    - capability flags allow omitting unsupported kwargs (e.g., o3).
    """

    def __init__(
        self,
        client,
        model: str,

    ):
        self.client = client
        self.model = model
        # Token usage tracking
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._num_calls: int = 0

    def _segments_to_content(self, segs: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for kind, val in segs:
            if kind == "text":
                if str(val).strip():
                    content.append({"type": "text", "text": str(val)})
            else:
                content.append({"type": "image_url", "image_url": {"url": pil_to_dataurl_png(val)}})
        return content

    def format_system(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "system", "content": self._segments_to_content(segs)}

    def format_user_turn(self, text: str, images: List[Image.Image]) -> Dict[str, Any]:
        segs = compile_text_images_for_order(text, images)
        return {"role": "user", "content": self._segments_to_content(segs)}

    def get_token_usage(self) -> Dict[str, Any]:
        """Return accumulated token usage stats for this adapter instance."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "num_calls": self._num_calls,
        }

    async def acompletion(self, messages: List[Dict[str, Any]], **chat_config: Any) -> str:

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **chat_config,
        )
        # Track token usage
        if resp.usage is not None:
            self._prompt_tokens += resp.usage.prompt_tokens or 0
            self._completion_tokens += resp.usage.completion_tokens or 0
            self._total_tokens += resp.usage.total_tokens or 0
            self._num_calls += 1
        return resp.choices[0].message.content or ""
