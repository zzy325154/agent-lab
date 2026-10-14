"""请求与响应模型。

用 Pydantic 做参数校验：非法输入在进入业务逻辑前就被挡住，
不会让一个空字符串或超长历史跑到 LLM 调用里浪费配额。
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="角色：user 或 assistant")
    content: str = Field(description="消息内容")


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="用户输入",
    )
    session_id: str | None = Field(
        default=None,
        description="会话 ID；不传则服务端新建一个，在响应中返回",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="客户端自带的历史（有会话 ID 时可省略）",
    )
    use_mcp: bool = Field(
        default=False,
        description="是否让工具经由 MCP Server 提供",
    )


class ToolCallSummary(BaseModel):
    name: str
    succeeded: bool
    attempts: int
    elapsed_ms: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rounds: int
    tool_calls: list[ToolCallSummary]
    tool_success_rate: float
    input_tokens: int
    output_tokens: int
    saved_pct: float


class SessionInfo(BaseModel):
    session_id: str
    turns: int
    messages: int


class HealthResponse(BaseModel):
    status: str
    provider: str
    local_tools: list[str]
    mcp_available: bool
