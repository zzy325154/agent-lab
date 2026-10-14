"""LLM 客户端抽象层。

把「模型厂商」和「Agent 循环」隔离开：Agent 只认 create_message() 这个接口，
换成 DeepSeek 还是 Anthropic 都不用改循环里的任何一行代码。

两家 API 的差异主要在两处，都在这层抹平：
  1. 工具 Schema 的字段名不同（input_schema vs parameters 且需包一层 function）
  2. 消息格式不同（原生 content block vs OpenAI 风格 tool_calls）
内部统一采用 Anthropic 风格的 content block 表示，因为它对
「一次返回多个工具调用」的表达最自然。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载器：只处理 KEY=VALUE 和 # 注释，已存在的环境变量优先。"""
    env_file = path or _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw_content: Any          # 要原样塞回 messages 的 assistant 内容
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class LLMClient:
    """所有厂商客户端的共同接口。"""

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    async def stream_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        """流式返回 (text_delta, final_response) 事件的异步生成器。

        默认实现回退到非流式：一次性产出完整文本再结束。
        这样 Agent 只需认这一个接口，不支持流式的后端也能跑。

        text_delta 是给用户看的增量文本，可能为空字符串（表示这一片没有正文）。
        final_response 是整轮的完整 LLMResponse，只会在最后 yield 一次。
        """
        response = await self.create_message(system, messages, tools)
        if response.text:
            yield ("text_delta", response.text)
        yield ("final", response)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise RuntimeError("请先安装: pip install anthropic") from e

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("缺少 ANTHROPIC_API_KEY 环境变量")
        self._client = AsyncAnthropic(api_key=key)
        self.model = model

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        resp = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                )

        # 原样保留，作为 assistant 轮次回填历史
        raw_content = [b.model_dump() for b in resp.content]

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            raw_content=raw_content,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason or "",
        )


# --------------------------------------------------------------------------
# DeepSeek（OpenAI 兼容协议）
# --------------------------------------------------------------------------


class DeepSeekClient(LLMClient):
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError("请先安装: pip install openai") from e

        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量")
        self._client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.model = model

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        oai_messages = _to_openai_messages(system, messages)

        kwargs: dict[str, Any] = {"model": self.model, "messages": oai_messages}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # 模型偶尔吐出非法 JSON，包成特殊参数让工具层报错回灌
                args = {"__parse_error__": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        # 转回 Anthropic 风格的 content block，保持下游统一
        raw_content: list[dict] = []
        if msg.content:
            raw_content.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            raw_content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments or "{}")
                    if _is_valid_json(tc.function.arguments)
                    else {},
                }
            )

        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            raw_content=raw_content,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=choice.finish_reason or "",
        )


    async def stream_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        oai_messages = _to_openai_messages(system, messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "stream": True,
            # 流式下若不显式要求，多数服务不返回 usage，token 统计就没了
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        stream = await self._client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        # 工具调用是分片到达的，按 index 累积
        acc: dict[int, dict[str, str]] = {}
        finish_reason = ""
        usage = None

        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta is None:
                continue

            if delta.content:
                text_parts.append(delta.content)
                yield ("text_delta", delta.content)

            for tc in delta.tool_calls or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        text = "".join(text_parts)
        yield ("final", self._assemble(text, acc, finish_reason, usage))

    def _assemble(
        self,
        text: str,
        acc: dict[int, dict[str, str]],
        finish_reason: str,
        usage: Any,
    ) -> LLMResponse:
        """把流式累积的分片拼成与非流式一致的 LLMResponse。"""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []
        if text:
            raw_content.append({"type": "text", "text": text})

        for index in sorted(acc):
            slot = acc[index]
            raw_args = slot["arguments"] or "{}"
            if _is_valid_json(raw_args):
                args = json.loads(raw_args)
            else:
                # 模型偶尔吐出截断或非法的 JSON，交给工具层报错回灌
                args = {"__parse_error__": raw_args}

            tool_calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=args))
            raw_content.append(
                {
                    "type": "tool_use",
                    "id": slot["id"],
                    "name": slot["name"],
                    "input": args if _is_valid_json(raw_args) else {},
                }
            )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw_content=raw_content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=finish_reason,
        )


def _is_valid_json(s: str | None) -> bool:
    if not s:
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """把 Anthropic 风格的 content block 历史，翻译成 OpenAI 的扁平消息格式。"""
    out: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts, calls = [], []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"], ensure_ascii=False),
                            },
                        }
                    )
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
            }
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue

        # user 轮次里的 tool_result -> OpenAI 的 tool 消息
        texts = []
        for block in content:
            if block.get("type") == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    }
                )
            elif block.get("type") == "text":
                texts.append(block.get("text", ""))
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})

    return out


def build_client(provider: str | None = None) -> LLMClient:
    """按环境变量或显式参数选择客户端。"""
    provider = (provider or os.getenv("LLM_PROVIDER") or "").lower()

    if not provider:
        # 没指定就自动探测谁的 key 在
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        else:
            raise RuntimeError(
                "未找到任何 API Key。请设置 ANTHROPIC_API_KEY 或 DEEPSEEK_API_KEY，"
                "或在 .env 中指定 LLM_PROVIDER。"
            )

    if provider == "anthropic":
        return AnthropicClient()
    if provider == "deepseek":
        return DeepSeekClient()
    raise ValueError(f"不支持的 provider: {provider}")
