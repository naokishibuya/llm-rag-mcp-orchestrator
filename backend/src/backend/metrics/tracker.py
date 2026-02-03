from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import threading

from ..config import calc_cost


@dataclass
class RequestMetrics:
    request_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    operation: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._requests: list[RequestMetrics] = []
        self._lock = threading.Lock()
        self._request_counter = 0

    def _next_id(self) -> str:
        self._request_counter += 1
        return f"req_{self._request_counter:06d}"

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        operation: str = "chat",
        **metadata,
    ) -> RequestMetrics:
        if not self.enabled:
            return None

        cost = calc_cost(model, input_tokens, output_tokens)

        metrics = RequestMetrics(
            request_id=self._next_id(),
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            operation=operation,
            metadata=metadata,
        )

        with self._lock:
            self._requests.append(metrics)

        return metrics

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_input = sum(r.input_tokens for r in self._requests)
            total_output = sum(r.output_tokens for r in self._requests)
            total_cost = sum(r.cost for r in self._requests)
            total_requests = len(self._requests)

        return {
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
        }

    def get_requests(self, limit: int = 100) -> list[RequestMetrics]:
        with self._lock:
            return list(self._requests[-limit:])

    def clear(self):
        with self._lock:
            self._requests.clear()
            self._request_counter = 0


_tracker: MetricsTracker | None = None


def get_tracker() -> MetricsTracker:
    global _tracker
    if _tracker is None:
        from ..config import get_metrics_config
        cfg = get_metrics_config()
        _tracker = MetricsTracker(enabled=cfg.get("enabled", True))
    return _tracker
