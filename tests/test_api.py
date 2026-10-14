"""HTTP API 测试。

用 FastAPI 的 TestClient 走完整请求链路（含 lifespan、校验、SSE 解析）。
只有 SSE 的流式连续性需要真实网络，那部分由 tests/smoke_api.py 覆盖。

运行：python tests/test_api.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.agent import Agent
from app.api_models import ChatRequest
from app.llm import LLMClient, LLMResponse, ToolCall
from app.server import app, store, _build_agent, _sse, _summarize


class ScriptedLLM(LLMClient):
    """脚本化模型，避免测试依赖真实 API 与余额。"""

    def __init__(self, script):
        self.script = list(script)
        self.seen_tools = []

    async def create_message(self, system, messages, tools=None):
        self.seen_tools.append(tools)
        if not self.script:
            return LLMResponse(
                text="（脚本已耗尽）",
                tool_calls=[],
                raw_content=[{"type": "text", "text": "（脚本已耗尽）"}],
            )
        return self.script.pop(0)

    async def stream_message(self, system, messages, tools=None):
        resp = await self.create_message(system, messages, tools)
        if resp.text:
            yield ("text_delta", resp.text)
        yield ("final", resp)


def _answer(text):
    return LLMResponse(
        text=text, tool_calls=[], raw_content=[{"type": "text", "text": text}]
    )


def _call(name, args, cid="c1"):
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(cid, name, args)],
        raw_content=[{"type": "tool_use", "id": cid, "name": name, "input": args}],
    )


# --------------------------------------------------------------------------
# 请求校验（不依赖模型，用 TestClient 直接打）
# --------------------------------------------------------------------------


def test_validation_rejects_empty_message():
    with TestClient(app) as client:
        r = client.post("/chat", json={"message": ""})
        assert r.status_code == 422, f"空消息应被拒，实际 {r.status_code}"
        assert "string_too_short" in r.text
    print("  [PASS] 空消息被 422 拒绝")


def test_validation_rejects_missing_field():
    with TestClient(app) as client:
        r = client.post("/chat", json={})
        assert r.status_code == 422
    print("  [PASS] 缺少 message 字段被拒")


def test_validation_rejects_overlong_message():
    with TestClient(app) as client:
        r = client.post("/chat", json={"message": "x" * 5000})
        assert r.status_code == 422, "超长输入应被拒"
        assert "string_too_long" in r.text
    print("  [PASS] 超长输入（>4000 字符）被拒")


def test_validation_rejects_too_many_history():
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={
                "message": "hi",
                "history": [{"role": "user", "content": "x"}] * 60,
            },
        )
        assert r.status_code == 422, "历史超长应被拒"
    print("  [PASS] 历史超过 50 条被拒")


def test_health_endpoint():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "calculator" in d["local_tools"]
        assert set(d["local_tools"]) == {
            "calculator",
            "get_current_time",
            "query_sales",
            "search_policy",
        }
    print(f"  [PASS] /health 正常（provider={d['provider']}, MCP={d['mcp_available']}）")


def test_tools_endpoint():
    with TestClient(app) as client:
        r = client.get("/tools")
        assert r.status_code == 200
        d = r.json()
        assert len(d["local"]) == 4
        names = {t["name"] for t in d["local"]}
        assert names == {"calculator", "get_current_time", "query_sales", "search_policy"}
        # 每个工具都该有参数描述，否则模型不知道参数含义
        for tool in d["local"]:
            props = tool["input_schema"]["properties"]
            for pname, p in props.items():
                assert "description" in p, f"{tool['name']}.{pname} 缺描述"
    print("  [PASS] /tools 返回 4 个工具且参数描述完整")


def test_unknown_session_404():
    with TestClient(app) as client:
        r = client.get("/sessions/nonexistent123")
        assert r.status_code == 404
    print("  [PASS] 不存在的会话返回 404")


# --------------------------------------------------------------------------
# SSE 帧构造
# --------------------------------------------------------------------------


def test_sse_frame_format():
    frame = _sse("text_delta", {"text": "你好\n世界"})
    assert frame.startswith("event: text_delta\n")
    assert frame.endswith("\n\n")
    # data 必须是单行 JSON，换行被转义
    data_line = [l for l in frame.split("\n") if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["text"] == "你好\n世界"
    assert "\n" not in data_line, "data 行内不应有裸换行，否则 SSE 解析会断"
    print("  [PASS] SSE 帧格式正确（多行内容被转义）")


# --------------------------------------------------------------------------
# 会话存储
# --------------------------------------------------------------------------


def test_session_persistence_and_eviction():
    import asyncio

    from app.session import SessionStore

    async def run():
        s = SessionStore(max_sessions=3)
        ids = []
        for i in range(5):
            sess = await s.get_or_create(None)
            ids.append(sess.session_id)
            await s.append(sess, f"q{i}", f"a{i}")

        # 上限 3，最早的应被淘汰
        assert s.count() == 3, f"会话数应为 3，实际 {s.count()}"
        assert await s.info(ids[0]) is None, "最早的会话应被淘汰"
        assert await s.info(ids[-1]) is not None, "最新的会话应保留"

        # 同一 session_id 应复用
        again = await s.get_or_create(ids[-1])
        assert again.session_id == ids[-1]
        assert again.turns == 1

    asyncio.run(run())
    print("  [PASS] 会话持久化与容量淘汰正确")


def test_session_lock_is_stable():
    """同一会话必须拿到同一把锁，否则并发请求会交错写入历史。"""
    import asyncio

    from app.session import SessionStore

    async def run():
        s = SessionStore()
        l1 = await s.lock_for("abc")
        l2 = await s.lock_for("abc")
        l3 = await s.lock_for("xyz")
        assert l1 is l2, "同一会话应返回同一把锁"
        assert l1 is not l3, "不同会话应使用不同的锁"

    asyncio.run(run())
    print("  [PASS] 会话锁按 session 隔离且稳定")


# --------------------------------------------------------------------------
# Agent 与 API 模型的衔接
# --------------------------------------------------------------------------


def test_summarize_maps_fields():
    import asyncio

    from app.registry import registry
    from app.tools import core

    llm = ScriptedLLM(
        [
            _call("calculator", {"expression": "6*7"}),
            _answer("42"),
        ]
    )
    agent = Agent(llm=llm, registry=registry, verbose=False)
    result = asyncio.run(agent.run("算 6*7"))

    summary = _summarize(result)
    assert summary.answer == "42"
    assert summary.rounds == 2
    assert len(summary.tool_calls) == 1
    assert summary.tool_calls[0].name == "calculator"
    assert summary.tool_success_rate == 100.0
    print("  [PASS] AgentResult -> ChatResponse 字段映射正确")


def test_build_agent_rejects_mcp_when_unavailable():
    """MCP 不可用时应明确报错，而不是静默降级到本地工具。"""
    from fastapi import HTTPException

    original = app.state.mcp
    original_err = app.state.mcp_error
    try:
        app.state.mcp = None
        app.state.mcp_error = "模拟启动失败"
        try:
            _build_agent(use_mcp=True)
            raise AssertionError("应抛出 HTTPException")
        except HTTPException as e:
            assert e.status_code == 503
            assert "模拟启动失败" in e.detail
    finally:
        app.state.mcp = original
        app.state.mcp_error = original_err
    print("  [PASS] MCP 不可用时返回 503 且带原因")


def main() -> int:
    tests = [
        test_validation_rejects_empty_message,
        test_validation_rejects_missing_field,
        test_validation_rejects_overlong_message,
        test_validation_rejects_too_many_history,
        test_health_endpoint,
        test_tools_endpoint,
        test_unknown_session_404,
        test_sse_frame_format,
        test_session_persistence_and_eviction,
        test_session_lock_is_stable,
        test_summarize_maps_fields,
        test_build_agent_rejects_mcp_when_unavailable,
    ]
    print("HTTP API 测试：")
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
