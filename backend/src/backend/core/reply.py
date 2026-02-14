from dataclasses import dataclass, field


@dataclass
class Tokens:
    input_tokens: int = 0
    output_tokens: int = 0

    def __str__(self) -> str:
        return f"tokens=[{self.input_tokens}/{self.output_tokens}]"


@dataclass
class Reply:
    text: str
    model: str = ""
    tokens: Tokens = field(default_factory=Tokens)
    tools_used: list[str] = field(default_factory=list)
    success: bool = True

    def __str__(self) -> str:
        parts = [f"[{self.model}] {self.tokens}: {self.text}"]
        if self.tools_used:
            parts.append(f"tools={self.tools_used}")
        if not self.success:
            parts.append("FAILED")
        return " ".join(parts)
