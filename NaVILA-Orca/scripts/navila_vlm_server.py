#!/usr/bin/env python3
"""Serve an installed NaVILA model through Orca_VLN's TCP protocol.

The server intentionally owns only the socket protocol and prompt boundary.
Model loading and image preprocessing remain in the separately installed
NaVILA package (``llava``).
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
from io import BytesIO

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images


MAX_REQUEST_BYTES = 256 * 1024 * 1024
HEALTH_RESPONSE = {
    "service": "navila-vlm",
    "status": "ok",
    "protocol_version": 1,
}


def recv_exact(connection: socket.socket, size: int) -> bytes | None:
    """Read one complete protocol field, returning ``None`` on early EOF."""

    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NaVILAServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path,
            model_name,
            args.model_base,
            device=args.device,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def infer(self, images: list[Image.Image], instruction: str) -> str:
        if len(images) != self.args.num_video_frames:
            raise ValueError(
                f"expected {self.args.num_video_frames} images, got {len(images)}"
            )

        image_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.args.device, dtype=torch.float16)
        conversation = conv_templates[self.args.conv_mode].copy()
        image_tokens = "<image>\n" * (self.args.num_video_frames - 1)
        prompt_text = (
            "Imagine you are a robot programmed for navigation tasks. You have "
            f"been given historical observations {image_tokens}and the current "
            f'observation <image>. Your assigned task is: "{instruction}". '
            "Analyze this series of images to decide the next action: turn left "
            "or right by a specific degree, move forward a specific distance, or "
            "stop when the task is complete."
        )
        conversation.append_message(conversation.roles[0], prompt_text)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        from llava.mm_utils import tokenizer_image_token

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.args.device)
        stop_string = (
            conversation.sep
            if conversation.sep_style != SeparatorStyle.TWO
            else conversation.sep2
        )
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_string], self.tokenizer, input_ids
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.host, self.args.port))
            server.listen(1)
            print(f"NaVILA VLM server listening on {self.args.host}:{self.args.port}")
            while True:
                connection, address = server.accept()
                with connection:
                    try:
                        self._handle_request(connection)
                    except Exception as exc:  # Keep the loaded model alive.
                        print(f"request from {address} failed: {type(exc).__name__}: {exc}")

    def _handle_request(self, connection: socket.socket) -> None:
        size_bytes = recv_exact(connection, 8)
        if size_bytes is None:
            return
        request_size = int.from_bytes(size_bytes, "big")
        if not 0 < request_size <= MAX_REQUEST_BYTES:
            raise ValueError(f"invalid request size: {request_size}")
        request_bytes = recv_exact(connection, request_size)
        if request_bytes is None:
            return
        request = json.loads(request_bytes.decode("utf-8"))
        if isinstance(request, dict) and request.get("type") == "health":
            response: object = HEALTH_RESPONSE
        else:
            images = [
                Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
                for encoded in request["images"]
            ]
            response = self.infer(images, str(request["query"]))
        response_bytes = json.dumps(response).encode("utf-8")
        connection.sendall(len(response_bytes).to_bytes(8, "big"))
        connection.sendall(response_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--model_base",
        default=None,
        help=(
            "base NaVILA model directory when model_path is an unmerged LoRA "
            "adapter directory"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv_mode", default="llama_3")
    parser.add_argument("--num_video_frames", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    NaVILAServer(parse_args()).serve_forever()
