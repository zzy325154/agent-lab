"""类型注解 -> JSON Schema 推导。

模型的 Function Calling 需要一份 JSON Schema 来描述工具入参。
手写 Schema 又臭又长还容易和函数签名对不上，这里直接从类型注解推导，
保证「文档」和「实现」永远一致。
"""

import inspect
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

_PRIMITIVES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def python_type_to_schema(annotation: Any) -> dict:
    # Annotated[X, Field(description=...)]：描述写在类型上，
    # 这样本地链路与 MCP 链路（走 Pydantic）都能读到同一份描述。
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        schema = python_type_to_schema(args[0])
        for meta in args[1:]:
            desc = getattr(meta, "description", None)
            if desc:
                schema["description"] = desc
        return schema

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        inner = python_type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": inner}

    # 兜底：当作字符串处理，不要因为一个没见过的类型让整个工具注册失败
    return {"type": "string"}


def _extract_param_doc(fn: Callable, param_name: str) -> str:
    """从 docstring 的 Args: 段落里抠出单个参数的说明。"""
    doc = fn.__doc__ or ""
    lines = doc.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        for prefix in (f"{param_name}:", f"{param_name} :"):
            if stripped.startswith(prefix):
                desc = stripped[len(prefix):].strip()
                # 说明可能折行，往下再看几行
                for follow in lines[i + 1 : i + 4]:
                    fs = follow.strip()
                    if not fs or ":" in fs.split(" ")[0]:
                        break
                    desc += " " + fs
                return desc
    return ""


def build_parameters_schema(fn: Callable) -> dict:
    # include_extras=True 是必需的：默认行为会把 Annotated 里的
    # Field 元数据剥掉，参数描述就全丢了。
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        prop = python_type_to_schema(annotation)
        # 类型上没写描述时，回退到 docstring 的 Args: 段落
        if "description" not in prop:
            param_doc = _extract_param_doc(fn, name)
            if param_doc:
                prop["description"] = param_doc
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}
