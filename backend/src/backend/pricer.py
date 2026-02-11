class Pricer:
    def __init__(self, pricing: dict):
        self._pricing = pricing

    def calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_pricing = self._pricing.get(model, {"input": 0.0, "output": 0.0})
        input_cost  = (input_tokens  / 1_000_000) * model_pricing.get("input", 0.0)
        output_cost = (output_tokens / 1_000_000) * model_pricing.get("output", 0.0)
        return input_cost + output_cost
