"""Loopback OpenAI-compatible multimodal serving on exactly two GPUs.

Batching is opt-in. The default max batch size is one until a pilot verifies the
installed Transformers/Qwen multimodal batch path. No remote model or image
fetches are permitted.
"""
import argparse
import base64
import binascii
import io
import json
import hashlib
import os
import threading
import time
import traceback
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 2048
MAX_BATCH_SIZE = 4
INLINE_IMAGE_PREFIXES = (
    "data:image/jpeg;base64,",
    "data:image/png;base64,",
)


class BatchCoordinator:
    """Group compatible work for one serialized generate_batch worker."""

    def __init__(
        self,
        generate_batch,
        compatibility_key,
        max_batch_size=1,
        gather_window_seconds=0.01,
    ):
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or not 1 <= max_batch_size <= MAX_BATCH_SIZE
        ):
            raise ValueError("max_batch_size must be an integer from 1 to 4")
        if gather_window_seconds < 0:
            raise ValueError("gather_window_seconds must be non-negative")
        self._generate_batch = generate_batch
        self._compatibility_key = compatibility_key
        self._max_batch_size = max_batch_size
        self._gather_window_seconds = gather_window_seconds
        self._condition = threading.Condition()
        self._pending = []
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="mr01-model-generation",
            daemon=True,
        )
        self._worker.start()

    @property
    def max_batch_size(self):
        return self._max_batch_size

    def submit(self, item):
        future = Future()
        with self._condition:
            if self._closed:
                future.set_exception(RuntimeError("batch coordinator is closed"))
                return future
            try:
                key = self._compatibility_key(item)
            except BaseException as exc:
                future.set_exception(exc)
                return future
            self._pending.append((item, future, key))
            self._condition.notify()
        return future

    def begin_shutdown(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending = self._pending
            self._pending = []
            self._condition.notify_all()
        error = RuntimeError("batch coordinator is shutting down")
        for unused_item, future, unused_key in pending:
            self._set_exception(future, error)

    def close(self):
        self.begin_shutdown()
        self._worker.join()

    def _next_batch(self):
        with self._condition:
            while not self._pending and not self._closed:
                self._condition.wait()
            if not self._pending:
                return None

            first = self._pending.pop(0)
            batch = [first]
            batch_key = first[2]
            if self._max_batch_size == 1:
                return batch

            deadline = time.monotonic() + self._gather_window_seconds
            while len(batch) < self._max_batch_size:
                match_index = next(
                    (
                        index
                        for index, (unused_item, unused_future, key)
                        in enumerate(self._pending)
                        if key == batch_key
                    ),
                    None,
                )
                if match_index is not None:
                    batch.append(self._pending.pop(match_index))
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed:
                    break
                self._condition.wait(remaining)
            return batch

    @staticmethod
    def _set_exception(future, error):
        if not future.cancelled() and not future.done():
            future.set_exception(error)

    @staticmethod
    def _set_result(future, result):
        if not future.cancelled() and not future.done():
            future.set_result(result)

    def _run(self):
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            items = [item for item, unused_future, unused_key in batch]
            futures = [
                future for unused_item, future, unused_key in batch
            ]
            try:
                results = list(self._generate_batch(items))
                if len(results) != len(items):
                    raise RuntimeError(
                        "generate_batch must return one result per request"
                    )
            except BaseException as exc:
                for future in futures:
                    self._set_exception(future, exc)
                continue

            for future, result in zip(futures, results):
                if isinstance(result, BaseException):
                    self._set_exception(future, result)
                else:
                    self._set_result(future, result)


def _require_int(value, field, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _decode_inline_image(url, image_loader):
    if not isinstance(url, str) or not url.startswith(INLINE_IMAGE_PREFIXES):
        raise ValueError("only inline image data allowed")
    encoded = url.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image data") from exc
    if not image_bytes:
        raise ValueError("invalid base64 image data: empty image")
    return image_loader(image_bytes)


def prepare_request(request, model_id, image_loader):
    """Validate and normalize one OpenAI-style request without model imports."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if request.get("model") != model_id:
        raise ValueError("wrong model identity")
    if request.get("stream", False) is not False:
        raise ValueError("stream must be false")

    temperature = request.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(
        temperature, (int, float)
    ) or temperature != 0:
        raise ValueError("temperature must be frozen at 0")
    max_tokens = _require_int(
        request.get("max_tokens", MAX_NEW_TOKENS),
        "max_tokens",
        minimum=1,
        maximum=MAX_NEW_TOKENS,
    )
    seed = _require_int(request.get("seed", 42), "seed")
    template_kwargs = request.get(
        "chat_template_kwargs", {"enable_thinking": False}
    )
    if not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    if template_kwargs.get("enable_thinking", False) is not False:
        raise ValueError("thinking must be disabled")

    source_messages = request.get("messages")
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError("messages must be a non-empty list")
    messages = []
    image_count = 0
    for message in source_messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("unsupported message role")
        content = message.get("content")
        if isinstance(content, str):
            normalized_content = content
        elif isinstance(content, list):
            normalized_content = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("content part must be an object")
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError("text content must be a string")
                    normalized_content.append({"type": "text", "text": text})
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if not isinstance(image_url, dict):
                        raise ValueError("image_url must be an object")
                    image = _decode_inline_image(
                        image_url.get("url"), image_loader
                    )
                    normalized_content.append(
                        {"type": "image", "image": image}
                    )
                    image_count += 1
                else:
                    raise ValueError("unsupported content type")
        else:
            raise ValueError("message content must be text or a list")
        messages.append({"role": role, "content": normalized_content})

    return {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": False,
        "image_count": image_count,
    }


def request_compatibility_key(request):
    """Parameters and modality shape that must agree within a batch."""
    return (
        request["model"],
        request["temperature"],
        request["max_tokens"],
        request["seed"],
        request["enable_thinking"],
        request["image_count"],
    )


def validate_context_lengths(
    prompt_token_counts,
    max_new_tokens,
    max_context_tokens=MAX_CONTEXT_TOKENS,
):
    for count in prompt_token_counts:
        if count + max_new_tokens > max_context_tokens:
            raise ValueError("context exceeds frozen 8192 cap")


def generated_row(
    token_ids,
    max_new_tokens,
    eos_token_ids,
    pad_token_id,
):
    """Trim one padded generation row and report its real token count."""
    eos_token_ids = set(eos_token_ids or ())
    generated = []
    stopped = False
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in eos_token_ids:
            generated.append(token_id)
            stopped = True
            break
        if pad_token_id is not None and token_id == pad_token_id:
            stopped = True
            break
        generated.append(token_id)
        if len(generated) == max_new_tokens:
            break
    count = len(generated)
    finish_reason = (
        "length"
        if count >= max_new_tokens and not stopped
        else "stop"
    )
    return generated, count, finish_reason


def validate_two_gpu_device_map(device_map):
    used = set()
    for value in device_map.values():
        if isinstance(value, bool):
            raise RuntimeError("model must be placed across exactly two GPUs")
        if isinstance(value, int):
            index = value
        elif isinstance(value, str):
            normalized = value.lower()
            if normalized in {"cpu", "disk"}:
                raise RuntimeError(
                    "model must use two GPUs without CPU/disk offload"
                )
            if normalized.startswith("cuda:"):
                normalized = normalized.split(":", 1)[1]
            if not normalized.isdigit():
                raise RuntimeError(
                    "model must be placed across exactly two GPUs"
                )
            index = int(normalized)
        else:
            raise RuntimeError(
                "model must be placed across exactly two GPUs"
            )
        used.add(index)
    if used != {0, 1}:
        raise RuntimeError(
            "model must be placed across exactly two GPUs"
        )
    return used


def server_build_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def health_payload(
    model_id,
    revision,
    run_id,
    visible_gpus,
    device_map,
    max_batch_size,
    instance_id=None,
    model_path=None,
    image_min_pixels=65536,
    image_max_pixels=262144,
):
    return {
        "ready": True,
        "model": model_id,
        "revision": revision,
        "run_id": run_id,
        "visible_gpus": visible_gpus,
        "placement": "balanced_two_gpu_bf16",
        "device_map": device_map,
        "max_batch_size": max_batch_size,
        "instance_id": instance_id,
        "model_path": model_path,
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "server_build_sha256": server_build_sha256(),
        "pid": os.getpid(),
    }


def validate_shutdown_payload(
    payload,
    expected_run_id,
    expected_instance_id=None,
):
    if not isinstance(payload, dict):
        raise ValueError("shutdown payload must contain run_id")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or run_id != expected_run_id:
        raise ValueError("shutdown run_id does not match this server")
    if (
        expected_instance_id is not None
        and payload.get("instance_id") != expected_instance_id
    ):
        raise ValueError("shutdown instance_id does not match this server")
    return run_id


def _token_ids(value):
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def build_generate_batch(torch, processor, model):
    """Build the only function allowed to call model.generate."""
    tokenizer = processor.tokenizer
    eos_token_ids = _token_ids(
        getattr(model.generation_config, "eos_token_id", None)
    )
    if not eos_token_ids:
        eos_token_ids = _token_ids(getattr(tokenizer, "eos_token_id", None))
    pad_token_id = getattr(model.generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def generate_batch(requests):
        if not requests:
            return []
        key = request_compatibility_key(requests[0])
        if any(request_compatibility_key(item) != key for item in requests):
            raise ValueError("incompatible requests reached model batch")

        messages_batch = [item["messages"] for item in requests]
        max_new_tokens = requests[0]["max_tokens"]
        seed = requests[0]["seed"]
        previous_padding_side = getattr(tokenizer, "padding_side", None)
        tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "padding": True,
                    "padding_side": "left",
                },
                enable_thinking=False,
            )
        finally:
            if previous_padding_side is not None:
                tokenizer.padding_side = previous_padding_side

        prompt_counts = [
            int(value)
            for value in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        validate_context_lengths(prompt_counts, max_new_tokens)
        padded_prompt_width = int(inputs["input_ids"].shape[1])
        inputs = inputs.to("cuda:0")
        torch.manual_seed(seed)
        started = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.time() - started

        results = []
        row_logs = []
        for index, request in enumerate(requests):
            padded_generated = outputs[index, padded_prompt_width:]
            if hasattr(padded_generated, "detach"):
                padded_generated = padded_generated.detach().cpu().tolist()
            else:
                padded_generated = list(padded_generated)
            tokens, count, finish_reason = generated_row(
                padded_generated,
                max_new_tokens,
                eos_token_ids,
                pad_token_id,
            )
            text = processor.decode(tokens, skip_special_tokens=True)
            row_logs.append(
                {
                    "index": index,
                    "input_tokens": prompt_counts[index],
                    "output_tokens": count,
                    "finish_reason": finish_reason,
                }
            )
            results.append(
                {
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_counts[index],
                        "completion_tokens": count,
                    },
                }
            )
        print(
            "BATCH",
            json.dumps(
                {
                    "batch_size": len(requests),
                    "seconds": elapsed,
                    "requests": row_logs,
                }
            ),
            flush=True,
        )
        return results

    return generate_batch


def make_handler(
    model_id,
    run_id,
    image_loader,
    coordinator,
    health,
    request_shutdown,
    instance_id=None,
):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(200, health)

        def _read_request_json(self):
            content_length = _require_int(
                int(self.headers["Content-Length"]),
                "Content-Length",
                minimum=1,
            )
            return json.loads(self.rfile.read(content_length))

        def do_POST(self):
            if self.path == "/shutdown":
                try:
                    payload = self._read_request_json()
                    validate_shutdown_payload(
                        payload, run_id, instance_id
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as exc:
                    self.send_json(
                        403,
                        {"error": type(exc).__name__ + ": " + str(exc)},
                    )
                    return
                coordinator.begin_shutdown()
                response = {
                    "status": "shutting_down",
                    "run_id": run_id,
                }
                if instance_id is not None:
                    response["instance_id"] = instance_id
                self.send_json(202, response)
                request_shutdown()
                return

            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            try:
                request = self._read_request_json()
                prepared = prepare_request(
                    request,
                    model_id,
                    image_loader,
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self.send_json(
                    400,
                    {"error": type(exc).__name__ + ": " + str(exc)},
                )
                return

            try:
                result = coordinator.submit(prepared).result()
                self.send_json(200, result)
            except BaseException as exc:
                traceback.print_exc()
                self.send_json(
                    500,
                    {"error": type(exc).__name__ + ": " + str(exc)},
                )

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Opt in after pilot; allowed range 1-4, default 1.",
    )
    parser.add_argument(
        "--batch-gather-ms",
        type=float,
        default=10.0,
        help="Short compatible-request gather window.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_batch_size <= MAX_BATCH_SIZE:
        parser.error("--max-batch-size must be from 1 to 4")
    if args.batch_gather_ms < 0:
        parser.error("--batch-gather-ms must be non-negative")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(4)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("exactly two visible GPUs required")
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_gpus is None or len(
        [item for item in visible_gpus.split(",") if item.strip()]
    ) != 2:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must identify exactly two GPUs"
        )

    start = time.time()
    print(
        "MODEL_LOADING",
        args.model_path,
        visible_gpus,
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "43GiB", 1: "43GiB"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    validate_two_gpu_device_map(model.hf_device_map)
    mapping = {
        str(key): str(value) for key, value in model.hf_device_map.items()
    }
    print(
        "MODEL_READY",
        json.dumps(
            {
                "load_seconds": time.time() - start,
                "device_map": mapping,
                "max_batch_size": args.max_batch_size,
            }
        ),
        flush=True,
    )

    def image_loader(image_bytes):
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    coordinator = BatchCoordinator(
        build_generate_batch(torch, processor, model),
        compatibility_key=request_compatibility_key,
        max_batch_size=args.max_batch_size,
        gather_window_seconds=args.batch_gather_ms / 1000.0,
    )
    health = health_payload(
        model_id=args.model_id,
        revision=args.revision,
        run_id=args.run_id,
        visible_gpus=visible_gpus,
        device_map=mapping,
        max_batch_size=args.max_batch_size,
        instance_id=args.instance_id,
        model_path=str(Path(args.model_path).resolve()),
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    server_holder = {}

    def request_shutdown():
        threading.Thread(
            target=server_holder["server"].shutdown,
            name="mr01-http-shutdown",
            daemon=True,
        ).start()

    handler = make_handler(
        model_id=args.model_id,
        run_id=args.run_id,
        image_loader=image_loader,
        coordinator=coordinator,
        health=health,
        request_shutdown=request_shutdown,
        instance_id=args.instance_id,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = False
    server_holder["server"] = server
    try:
        server.serve_forever()
    finally:
        server.server_close()
        coordinator.close()


if __name__ == "__main__":
    main()
