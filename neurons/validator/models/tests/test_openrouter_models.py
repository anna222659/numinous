from decimal import Decimal

from neurons.validator.models.chutes import ChatCompletionChoice, ChatCompletionMessage
from neurons.validator.models.openrouter import (
    DEFAULT_PRICING,
    OPENROUTER_MODEL_PRICING,
    OpenRouterCompletion,
    OpenRouterUsage,
    calculate_cost,
    estimate_cost_from_tokens,
)


class TestOpenRouterUsage:
    def test_usage_with_cost(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            cost=Decimal("0.00165"),
        )
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 100
        assert usage.total_tokens == 150
        assert usage.cost == Decimal("0.00165")

    def test_usage_without_cost(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )
        assert usage.cost is None

    def test_usage_allows_extra_fields(self):
        usage = OpenRouterUsage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            some_extra_field="value",
        )
        assert usage.model_extra["some_extra_field"] == "value"


class TestOpenRouterCompletion:
    def _make_choice(self, content: str = "Response") -> ChatCompletionChoice:
        return ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
            finish_reason="stop",
        )

    def test_completion_minimal(self):
        completion = OpenRouterCompletion(
            id="gen-123",
            created=1709000000,
            model="anthropic/claude-sonnet-4-6",
            choices=[self._make_choice()],
        )
        assert completion.id == "gen-123"
        assert completion.model == "anthropic/claude-sonnet-4-6"
        assert len(completion.choices) == 1
        assert completion.usage is None

    def test_completion_with_usage(self):
        completion = OpenRouterCompletion(
            id="gen-456",
            created=1709000000,
            model="google/gemini-2.5-flash",
            choices=[self._make_choice()],
            usage=OpenRouterUsage(
                prompt_tokens=30,
                completion_tokens=80,
                total_tokens=110,
                cost=Decimal("0.00053"),
            ),
        )
        assert completion.usage.prompt_tokens == 30
        assert completion.usage.cost == Decimal("0.00053")

    def test_completion_allows_extra_fields(self):
        completion = OpenRouterCompletion(
            id="gen-789",
            created=1709000000,
            model="test",
            choices=[self._make_choice()],
            system_fingerprint="abc",
        )
        assert completion.model_extra["system_fingerprint"] == "abc"


class TestModelPricing:
    def test_known_models_exist(self):
        assert "anthropic/claude-sonnet-4-6" in OPENROUTER_MODEL_PRICING
        assert "anthropic/claude-opus-4-6" in OPENROUTER_MODEL_PRICING
        assert "anthropic/claude-haiku-4-5" in OPENROUTER_MODEL_PRICING
        assert "google/gemini-2.5-pro" in OPENROUTER_MODEL_PRICING
        assert "google/gemini-2.5-flash" in OPENROUTER_MODEL_PRICING

    def test_pricing_format(self):
        for model, (input_cost, output_cost) in OPENROUTER_MODEL_PRICING.items():
            assert isinstance(input_cost, float), f"{model} input_cost not float"
            assert isinstance(output_cost, float), f"{model} output_cost not float"
            assert input_cost > 0, f"{model} input_cost not positive"
            assert output_cost > 0, f"{model} output_cost not positive"

    def test_default_pricing(self):
        assert DEFAULT_PRICING == (3.0, 15.0)


class TestCalculateCost:
    def _make_completion(self, usage=None) -> OpenRouterCompletion:
        return OpenRouterCompletion(
            id="gen-test",
            created=1709000000,
            model="test",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def test_cost_from_usage(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
                cost=Decimal("0.0045"),
            )
        )
        assert calculate_cost(completion) == Decimal("0.0045")

    def test_cost_zero_when_usage_cost_is_none(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
            )
        )
        assert calculate_cost(completion) == Decimal("0")

    def test_cost_zero_when_no_usage(self):
        completion = self._make_completion()
        assert calculate_cost(completion) == Decimal("0")

    def test_cost_zero_value(self):
        completion = self._make_completion(
            usage=OpenRouterUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=Decimal("0"),
            )
        )
        assert calculate_cost(completion) == Decimal("0")


class TestEstimateCostFromTokens:
    def test_known_model(self):
        cost = estimate_cost_from_tokens("anthropic/claude-sonnet-4-6", 1000, 500)
        expected = Decimal(str((3.0 * 1000 + 15.0 * 500) / 1_000_000))
        assert cost == expected

    def test_unknown_model_uses_default(self):
        cost = estimate_cost_from_tokens("unknown/model", 1000, 500)
        expected = Decimal(str((3.0 * 1000 + 15.0 * 500) / 1_000_000))
        assert cost == expected

    def test_zero_tokens(self):
        cost = estimate_cost_from_tokens("anthropic/claude-sonnet-4-6", 0, 0)
        assert cost == Decimal("0")

    def test_cheap_model(self):
        cost = estimate_cost_from_tokens("google/gemini-2.5-flash-lite", 1_000_000, 1_000_000)
        assert cost == Decimal("0.375")

    def test_expensive_model(self):
        cost = estimate_cost_from_tokens("anthropic/claude-opus-4-6", 1000, 500)
        expected = Decimal(str((15.0 * 1000 + 75.0 * 500) / 1_000_000))
        assert cost == expected
