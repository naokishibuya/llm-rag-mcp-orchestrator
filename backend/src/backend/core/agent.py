from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .reply import Reply


@dataclass
class UserContext:
    city: str | None = None
    timezone: str | None = None
    local_time: str | None = None

    def __bool__(self) -> bool:
        return any(v is not None for v in asdict(self).values())

    def __str__(self) -> str:
        return f"user_context={asdict(self)}"


class Agent(Protocol):
    async def act(self, *, query: str, context: UserContext, **kwargs: Any) -> Reply: ...
