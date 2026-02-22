from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol


class WebSocketFrameStream:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._clients: set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()
        self._commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        async def handler(ws: WebSocketServerProtocol) -> None:
            async with self._lock:
                self._clients.add(ws)
            try:
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        await self._commands.put(data)
            finally:
                async with self._lock:
                    self._clients.discard(ws)

        async with websockets.serve(handler, self.host, self.port):
            await asyncio.Future()

    async def publish(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload)
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        await asyncio.gather(*(c.send(msg) for c in clients), return_exceptions=True)

    def drain_commands(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except asyncio.QueueEmpty:
                return out
