"""FastAPI 服务：把 Agent 能力对外提供。

两个接口：
    POST /chat        一次性返回完整结果
    POST /chat/stream SSE 流式返回过程事件（模型正文增量 + 工具调用进度）

设计上刻意让两个接口共用同一套 Agent 与 session 逻辑，
差别只在「结果何时交给调用方」——避免流式与非流式出现行为分叉。

启动：python -m app.server
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import Agent
from app.api_models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionInfo,
    ToolCallSummary,
)
from app.llm import build_client
from app.mcp_client import MCPClient
from app.registry import registry
from app.session import SessionStore

# 导入即注册本地工具
import app.tools.builtin  # noqa: F401

store = SessionStore()
_started_at = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热：拉起常驻 MCP 会话 + 构建 RAG 索引。

    两件事都属于「冷启动成本」，放在启动阶段做掉，
    避免第一个用户请求承担几十秒的初始化。
    """
    app.state.mcp = None
    app.state.mcp_error = None
    app.state.rag_seconds = 0.0
    app.state.rag_error = None

    try:
        from app.rag import service as rag_service

        app.state.rag_seconds = await rag_service.warmup()
    except Exception as e:  # noqa: BLE001 — 检索不可用不该让服务起不来
        app.state.rag_error = f"{type(e).__name__}: {e}"

    try:
        mcp = MCPClient()
        await mcp.start(timeout=30.0)
        app.state.mcp = mcp
    except Exception as e:  # noqa: BLE001 — MCP 不可用不该让服务起不来
        app.state.mcp_error = f"{type(e).__name__}: {e}"

    yield

    if app.state.mcp is not None:
        await app.state.mcp.stop()


app = FastAPI(
    title="Agent Lab API",
    version="1.0.0",
    description="工具调用型 Agent 的 HTTP 接口",
    lifespan=lifespan,
)


def _build_agent(use_mcp: bool) -> Agent:
    mcp = app.state.mcp if use_mcp else None
    if use_mcp and mcp is None:
        raise HTTPException(
            status_code=503,
            detail=f"MCP 会话不可用：{app.state.mcp_error or '未启动'}",
        )
    return Agent(
        llm=build_client(),
        registry=None if use_mcp else registry,
        mcp=mcp,
        verbose=False,
    )


def _summarize(result) -> ChatResponse:  # type: ignore[no-untyped-def]
    return ChatResponse(
        session_id="",  # 由调用方填充
        answer=result.answer,
        rounds=result.rounds,
        tool_calls=[
            ToolCallSummary(
                name=r.tool_name,
                succeeded=r.succeeded,
                attempts=r.attempts,
                elapsed_ms=round(r.elapsed_ms, 1),
            )
            for r in result.tool_calls
        ],
        tool_success_rate=round(result.tool_success_rate, 1),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        saved_pct=round(result.saved_pct, 1),
    )


# --------------------------------------------------------------------------
# 一次性接口
# --------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session = await store.get_or_create(req.session_id)
    lock = await store.lock_for(session.session_id)

    async with lock:
        agent = _build_agent(req.use_mcp)
        history = (
            [m.model_dump() for m in req.history]
            if req.history
            else await store.history(session)
        )
        if req.history:
            # 客户端自己带了历史，以它为准并重置服务端会话
            session.messages = list(history)

        result = await agent.run(req.message, history=history)
        await store.append(session, req.message, result.answer)

    resp = _summarize(result)
    resp.session_id = session.session_id
    return resp


# --------------------------------------------------------------------------
# 流式接口（SSE）
# --------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """SSE 帧格式。data 里可能有换行，必须整体序列化成一行 JSON。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """以 SSE 流式返回。

    事件序列形如：
        round_start -> text_delta* -> (tool_start -> tool_end)*
        -> round_end -> ... -> done

    之所以用 POST + StreamingResponse 而不是 WebSocket：
    对话是「一问一答」的请求-响模型，SSE 更贴合且能复用 HTTP 基础设施。
    """
    session = await store.get_or_create(req.session_id)

    async def event_source():
        lock = await store.lock_for(session.session_id)
        async with lock:
            # 先告诉客户端会话 ID，后续多轮对话要带上
            yield _sse("session", {"session_id": session.session_id})

            agent = _build_agent(req.use_mcp)
            history = (
                [m.model_dump() for m in req.history]
                if req.history
                else await store.history(session)
            )
            if req.history:
                session.messages = list(history)

            answer_parts: list[str] = []

            try:
                async for kind, payload in agent.stream(
                    req.message, history=history
                ):
                    if kind == "text_delta":
                        answer_parts.append(payload["text"])
                        yield _sse("text_delta", payload)
                    elif kind == "done":
                        result = payload
                        await store.append(session, req.message, result.answer)
                        summary = _summarize(result)
                        summary.session_id = session.session_id
                        yield _sse("done", summary.model_dump())
                    elif kind == "error":
                        yield _sse("error", payload)
                    else:
                        # round_start / tool_start / tool_end / round_end
                        # / context_compressed 原样透传
                        yield _sse(kind, payload)
            except asyncio.CancelledError:
                # 客户端断开：保留已产生的部分回答，避免丢失
                if answer_parts:
                    await store.append(
                        session, req.message, "".join(answer_parts) + "（已中断）"
                    )
                raise
            except Exception as e:  # noqa: BLE001
                yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 Nginx 缓冲，否则流式会被攒成一次性返回
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------
# 辅助接口
# --------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    provider = "unknown"
    try:
        provider = type(build_client()).__name__
    except Exception:  # noqa: BLE001 — 缺 key 时仍应能探活
        provider = "unconfigured"

    return HealthResponse(
        status="ok",
        provider=provider,
        local_tools=registry.names(),
        mcp_available=app.state.mcp is not None,
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str) -> SessionInfo:
    info = await store.info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return info


@app.get("/tools")
async def list_tools() -> dict:
    """列出当前可用的工具，便于前端展示与排查。"""
    return {
        "local": registry.all_schemas(),
        "mcp": app.state.mcp.all_schemas() if app.state.mcp else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
