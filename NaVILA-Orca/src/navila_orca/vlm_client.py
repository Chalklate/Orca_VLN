"""Client for the original NaVILA 8-byte length-prefixed JSON protocol."""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence

from PIL import Image

from .frames import NUM_VIDEO_FRAMES, encode_images_jpeg_base64


class VLMProtocolError(RuntimeError):
    """Raised when a peer violates NaVILA's TCP protocol."""


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Receive exactly ``size`` bytes or fail on an early EOF."""

    if size < 0:
        raise ValueError("size must be non-negative")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            received = size - remaining
            raise VLMProtocolError(
                f"connection closed after {received} of {size} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LengthPrefixedJsonVLMClient:
    """Synchronous client for the bundled NavVLM TCP inference server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 54321,
        *,
        timeout_s: float = 120.0,
        max_response_bytes: int = 1 << 20,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)

    def infer(self, images: Sequence[Image.Image], instruction: str) -> str:
        if len(images) != NUM_VIDEO_FRAMES:
            raise ValueError(
                f"VLM inference requires exactly {NUM_VIDEO_FRAMES} images, got {len(images)}"
            )
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")

        request = {
            "images": encode_images_jpeg_base64(images),
            "query": instruction,
        }
        request_bytes = json.dumps(request).encode("utf-8")

        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout_s
            )
        except ConnectionRefusedError as exc:
            raise ConnectionError(
                "NaVILA VLM server is not listening at "
                f"{self.host}:{self.port}; start scripts/start_navvlm_server.sh "
                "and bind it to a reachable address first"
            ) from exc
        with sock:
            sock.settimeout(self.timeout_s)
            sock.sendall(len(request_bytes).to_bytes(8, "big"))
            sock.sendall(request_bytes)

            response_size = int.from_bytes(recv_exact(sock, 8), "big")
            if response_size <= 0 or response_size > self.max_response_bytes:
                raise VLMProtocolError(
                    f"invalid response size {response_size}; maximum is {self.max_response_bytes}"
                )
            response_bytes = recv_exact(sock, response_size)

        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VLMProtocolError("response is not valid UTF-8 JSON") from exc
        if not isinstance(response, str) or not response.strip():
            raise VLMProtocolError("response JSON must be a non-empty string")
        return response
