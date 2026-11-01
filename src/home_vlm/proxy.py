"""Forward 127.0.0.1:8700 on the host to the app on the internal network.

The app and model containers have no route out. Docker can only publish a port
from a network that has one, so this is the one container on both: it accepts
inbound connections and relays bytes, and starts nothing outbound itself.
"""

from __future__ import annotations

import asyncio
import os


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def main(listen_port: int, target_host: str, target_port: int) -> None:
    async def handle(client_reader, client_writer):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(_pipe(client_reader, upstream_writer), _pipe(upstream_reader, client_writer))

    server = await asyncio.start_server(handle, "0.0.0.0", listen_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(os.environ.get("LISTEN_PORT", "8700")), os.environ["TARGET_HOST"], int(os.environ["TARGET_PORT"])))
