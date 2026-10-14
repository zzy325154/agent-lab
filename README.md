# Agent Lab

工具调用型智能体系统 —— 覆盖「检索 → 规划 → 工具调用 → 再决策」完整闭环，工具通过 MCP 协议独立成 Server 进程，对外提供 HTTP + SSE 服务。

核心实现**不依赖 LangChain 等框架**，Agent 循环、上下文管理、错误恢复、检索链路均为手写，以便理解底层机制。

## 功能

- **MCP 工具接入**：工具以独立进程按标准协议暴露，本地直调与 MCP 两种接入方式共用同一份实现
- **RAG 检索**：语义分块 + 向量检索与 BM25 混合召回 + RRF 融合
- **并发调度**：无依赖的工具调用并行执行，结果按原顺序回灌
- **上下文管理**：两级压缩（工具结果裁剪 + 历史轮次裁剪），保证 tool_use/tool_result 成对
- **错误恢复**：按故障类型选择性重试、指数退避、熔断降级
- **服务化**：FastAPI + SSE 流式输出，进程内会话管理

## 快速开始

```bash
pip install -r requirements.txt

# 配置 API Key（DeepSeek 或 Anthropic）
echo "LLM_PROVIDER=deepseek" > .env
echo "DEEPSEEK_API_KEY=你的key" >> .env

# 命令行交互
python -m app.main

# 启动 HTTP 服务（含 RAG 索引预热）
python -m app.server
```

首次运行 RAG 检索会自动下载向量模型（约 95MB，走 hf-mirror 镜像）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 一次性返回完整结果 |
| POST | `/chat/stream` | SSE 流式返回正文与工具调用进度 |
| GET | `/health` | 健康检查 |

## 测试

```bash
for t in test_agent_loop test_context test_resilience test_mcp_integration test_api test_rag; do
  python tests/$t.py
done
```

共 47 项测试，覆盖 Agent 循环、上下文管理、错误恢复、MCP 协议、HTTP 接口与检索链路。检索效果测量见 `tests/measure_rag.py`，成功率测量见 `tests/measure_reliability.py`。

## 结构

```
app/
  agent.py         Agent 核心循环（含流式事件）
  llm.py           LLM 客户端（DeepSeek / Anthropic 双后端）
  registry.py      工具注册中心
  resilience.py    超时 / 重试 / 熔断
  context.py       上下文管理与 token 估算
  mcp_server.py    MCP Server
  mcp_client.py    MCP 客户端适配器
  server.py        FastAPI 服务
  session.py       会话存储
  tools/core.py    工具实现（唯一来源）
  rag/             检索链路（分块 / 向量 / BM25 / 融合）
```
