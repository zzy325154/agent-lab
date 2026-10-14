"""会话存储。

刻意先从内存实现：单进程、够用、零依赖。
换成 Redis 只需实现同样三个方法，业务代码不用改——
这也是简历里不该写 Redis 的原因（还没做）。

注意并发安全：FastAPI 的请求是并发处理的，同一个 session 可能被
两个请求同时读写，因此所有变更都放在 asyncio.Lock 里。
"""

import asyncio
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    turns: int = 0


class SessionStore:
    """进程内的会话存储，带容量上限防止无限增长。"""

    def __init__(self, max_sessions: int = 200) -> None:
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._max = max_sessions
        self._global_lock = asyncio.Lock()

    async def lock_for(self, session_id: str) -> asyncio.Lock:
        """取该会话的锁。同一会话的请求会被串行化，避免历史交错。"""
        async with self._global_lock:
            return self._locks.setdefault(session_id, asyncio.Lock())

    async def get_or_create(self, session_id: str | None) -> Session:
        async with self._global_lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            new_id = session_id or _new_id()
            if len(self._sessions) >= self._max:
                # 简单淘汰：丢掉最早插入的一个
                oldest = next(iter(self._sessions))
                self._sessions.pop(oldest, None)
                self._locks.pop(oldest, None)

            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    async def append(self, session: Session, user_input: str, answer: str) -> None:
        session.messages.append({"role": "user", "content": user_input})
        session.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
        session.turns += 1

    async def history(self, session: Session) -> list[dict]:
        return list(session.messages)

    async def info(self, session_id: str) -> SessionInfo | None:  # type: ignore[name-defined]
        session = self._sessions.get(session_id)
        if session is None:
            return None
        from app.api_models import SessionInfo

        return SessionInfo(
            session_id=session.session_id,
            turns=session.turns,
            messages=len(session.messages),
        )

    def count(self) -> int:
        return len(self._sessions)


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]
