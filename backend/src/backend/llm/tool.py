import inspect


def tool(fn):
    """Decorator that attaches a `.tool_schema` dict derived from the function signature."""
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        prop: dict = {"type": "string"}
        if param.annotation is float:
            prop["type"] = "number"
        elif param.annotation is int:
            prop["type"] = "integer"
        elif param.annotation is bool:
            prop["type"] = "boolean"
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    fn.tool_schema = {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }
    return fn
