from ..core import Tokens


class Pricer:
    def __init__(self, pricing: dict):
        self._pricing = pricing
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    def add(self, model: str, tokens: Tokens) -> dict:
        cost = self._calc_cost(model, tokens)
        self._total_input += tokens.input_tokens
        self._total_output += tokens.output_tokens
        self._total_cost += cost
        return {"input_tokens": tokens.input_tokens, "output_tokens": tokens.output_tokens, "cost": round(cost, 6)}

    def summary(self) -> dict:
        return {"total": {
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "cost": round(self._total_cost, 6),
        }}

    def _calc_cost(self, model: str, tokens: Tokens) -> float:
        model_pricing = self._pricing.get(model, {"input": 0.0, "output": 0.0})
        input_cost  = (tokens.input_tokens  / 1_000_000) * model_pricing.get("input", 0.0)
        output_cost = (tokens.output_tokens / 1_000_000) * model_pricing.get("output", 0.0)
        return input_cost + output_cost
