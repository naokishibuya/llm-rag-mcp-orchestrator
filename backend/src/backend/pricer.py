from .orchestrator.state import TokenUsage


class Pricer:
    def __init__(self, pricing: dict):
        self._pricing = pricing
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0

    def add(self, entries: list[TokenUsage]) -> dict | None:
        if not entries:
            return None
        step_input = 0
        step_output = 0
        step_cost = 0.0
        for entry in entries:
            cost = self._calc_cost(entry)
            step_input += entry.input_tokens
            step_output += entry.output_tokens
            step_cost += cost
        self._total_input += step_input
        self._total_output += step_output
        self._total_cost += step_cost
        return {"input_tokens": step_input, "output_tokens": step_output, "cost": round(step_cost, 6)}

    def summary(self) -> dict:
        return {"total": {
            "input_tokens": self._total_input,
            "output_tokens": self._total_output,
            "cost": round(self._total_cost, 6),
        }}

    def _calc_cost(self, entry: TokenUsage) -> float:
        model_pricing = self._pricing.get(entry.model, {"input": 0.0, "output": 0.0})
        input_cost  = (entry.input_tokens  / 1_000_000) * model_pricing.get("input", 0.0)
        output_cost = (entry.output_tokens / 1_000_000) * model_pricing.get("output", 0.0)
        return input_cost + output_cost
