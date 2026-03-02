from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from neurons.validator.models.chutes import ChatCompletionChoice


class OpenRouterUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[Decimal] = None

    model_config = ConfigDict(extra="allow")


class OpenRouterCompletion(BaseModel):
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[OpenRouterUsage] = None

    model_config = ConfigDict(extra="allow")


OPENROUTER_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4-6": (3.0, 15.0),
    "anthropic/claude-opus-4-6": (15.0, 75.0),
    "anthropic/claude-haiku-4-5": (0.8, 4.0),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-2.5-flash": (0.15, 0.60),
    "google/gemini-2.5-flash-lite": (0.075, 0.30),
}

DEFAULT_PRICING: tuple[float, float] = (3.0, 15.0)


def calculate_cost(completion: OpenRouterCompletion) -> Decimal:
    if completion.usage and completion.usage.cost is not None:
        return completion.usage.cost
    return Decimal("0")


def estimate_cost_from_tokens(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    input_cost_per_1m, output_cost_per_1m = OPENROUTER_MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (input_cost_per_1m * input_tokens + output_cost_per_1m * output_tokens) / 1_000_000
    return Decimal(str(cost))
