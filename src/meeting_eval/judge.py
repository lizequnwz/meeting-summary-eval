"""Bridge LangChain chat models to DeepEval's structured judge interface."""

from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel


def _first_string(model, attributes) -> str | None:
    for attribute in attributes:
        try:
            value = getattr(model, attribute, None)
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class LangChainJudge(DeepEvalBaseLLM):
    """Use a LangChain model without changing native prompts or score calculations."""

    def __init__(self, chat_model: BaseChatModel):
        if not isinstance(chat_model, BaseChatModel):
            raise TypeError("The judge must wrap a LangChain BaseChatModel.")
        self.chat_model = chat_model
        self.base_url = _first_string(chat_model, ("openai_api_base", "azure_endpoint", "base_url"))
        super().__init__(model=self.get_model_name())

    def load_model(self) -> BaseChatModel:
        return self.chat_model

    def get_model_name(self) -> str:
        return _first_string(self.chat_model, ("model_name", "model", "deployment_name")) or type(self.chat_model).__name__

    def _structured(self, schema):
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise ValueError("Judge calls require a Pydantic response schema.")
        return self.chat_model.with_structured_output(schema, include_raw=True)

    @staticmethod
    def _parsed(response, schema):
        if not isinstance(response, dict) or response.get("parsing_error") is not None:
            raise ValueError("Judge returned an invalid structured response.")
        raw, parsed = response.get("raw"), response.get("parsed")
        if not isinstance(raw, AIMessage) or not isinstance(parsed, schema):
            raise ValueError("Judge returned an invalid structured response.")
        refused = raw.additional_kwargs.get("refusal") or (
            isinstance(raw.content, list)
            and any(isinstance(block, dict) and block.get("type") == "refusal" for block in raw.content)
        )
        if refused:
            raise ValueError("Judge refused the structured response.")
        return schema.model_validate(parsed.model_dump())

    def generate(self, prompt: str, schema=None):
        response = self._structured(schema).invoke(prompt)
        return self._parsed(response, schema)

    async def a_generate(self, prompt: str, schema=None):
        response = await self._structured(schema).ainvoke(prompt)
        return self._parsed(response, schema)
