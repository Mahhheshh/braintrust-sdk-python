import time
from typing import Any

from braintrust.integrations.utils import _camel_to_snake, _resolved_attachment_from_bytes
from braintrust.logger import start_span
from braintrust.span_types import SpanTypeAttribute
from braintrust.util import merge_dicts


_RUNTIME_PROVIDER = "aws_bedrock_runtime"

_MIME_TYPES = {
    "document": {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "doc": "application/msword",
        "html": "text/html",
        "txt": "text/plain",
        "md": "text/markdown",
    },
    "image": {
        "gif": "image/gif",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    },
    "video": {
        "flv": "video/flv",
        "mkv": "video/x-matroska",
        "mov": "video/quicktime",
        "mp4": "video/mp4",
        "mpeg": "video/mpeg",
        "mpg": "video/mpeg",
        "three_gp": "video/3gpp",
        "webm": "video/webm",
        "wmv": "video/x-ms-wmv",
    },
    "audio": {
        "aac": "audio/aac",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "mka": "audio/x-matroska",
        "mkv": "audio/x-matroska",
        "mp3": "audio/mp3",
        "mp4": "audio/mp4",
        "mpeg": "audio/mpeg",
        "mpga": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/opus",
        "pcm": "audio/pcm",
        "wav": "audio/wav",
        "webm": "audio/webm",
        "x-aac": "audio/aac",
    },
}


def _normalize_converse_media_block(block_type: str, block: dict[str, Any]) -> dict[str, Any]:
    normalized_block = {key: value for key, value in block.items() if key != "source"}

    source = block.get("source")
    if not isinstance(source, dict):
        return normalized_block

    source_bytes = source.get("bytes")
    if source_bytes is not None:
        file_format = block.get("format", "unknown")
        mime_type = _MIME_TYPES.get(block_type, {}).get(file_format, "application/octet-stream")
        resolved_attachment = _resolved_attachment_from_bytes(
            source_bytes,
            mime_type=mime_type,
            prefix=block_type,
        )
        normalized_block["source"] = resolved_attachment.multimodal_part_payload
    else:
        normalized_block["source"] = source

    return normalized_block


def _normalize_converse_message_content(message_contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """helper function for `_get_converse_request`"""
    normalized_content: list[dict[str, Any]] = []
    for part_content in message_contents:
        if part_content.get("text") is not None:
            normalized_content.append({"text": part_content.get("text")})
            continue

        for block_type in ("image", "document", "video", "audio"):
            block = part_content.get(block_type)
            if isinstance(block, dict):
                normalized_content.append({block_type: _normalize_converse_media_block(block_type, block)})
                break
        else:
            normalized_content.append(part_content)

    return normalized_content


def _get_converse_input(**kwargs: dict[str, Any]) -> dict[str, Any]:
    """normalize raw input into braintrust supporeted Attachments for multimodel inputs"""
    input_data: dict[str, Any] = {}

    messages = kwargs.get("messages", None)
    if messages is not None:
        normalized_messages = []
        for message in messages:
            content = message.get("content")
            normalized_messages.append(
                {
                    **message,
                    "content": _normalize_converse_message_content(content),
                }
            )
        input_data["messages"] = normalized_messages

    return input_data


def _get_converse_request_metadata(**kwargs: dict[str, Any]):
    """Log metadata for boto3; track model id, request-id, system instructions, inference config"""
    model_id = kwargs.get("modelId", None)
    request_id = kwargs.get("amz-sdk-invocation-id")
    instructions = kwargs.get("system")
    inference_config = kwargs.get("inferenceConfig")

    metadata: dict[str, Any] = {}
    metadata["runtime_provider"] = _RUNTIME_PROVIDER

    if model_id is not None:
        model_details = model_id.split(".")
        if len(model_details) < 2:
            metadata["model"] = model_id
        else:
            model_provider, model_name = model_details
            metadata["model"] = model_name.split(":")[0]
            metadata["model_provider"] = model_provider

    if request_id is not None:
        metadata["request_id"] = request_id

    if instructions is not None:
        final_instruction = ""
        for instruction in instructions:
            insturction_chunk = instruction.get("text", None)
            if insturction_chunk is not None:
                final_instruction += insturction_chunk

        if final_instruction != "":
            metadata["instructions"] = final_instruction

    if inference_config is not None:
        inference_dict = {}
        for key, value in inference_config.items():
            inference_dict[_camel_to_snake(key)] = value
        merge_dicts(metadata, inference_dict)
    return metadata


def parse_converse_result(result: dict[str, Any]):
    """"""
    message = result.get("output", {}).get("message", {})
    content = message.get("content")

    _normalized_content = _normalize_converse_message_content(content)

    return {"message": {**message, "content": _normalized_content}}


def _extract_converse_metrics(response: dict[str, Any], start_time: float, end_time: float) -> dict[str, float]:
    """Extract metrics such as input tokens/output tokens and total tokens"""
    metrics: dict[str, float] = {}

    metrics["start"] = start_time
    metrics["end"] = end_time
    metrics["duration"] = end_time - start_time

    usage: dict[str, float] | None = response.get("usage", None)

    if usage is not None:
        if usage.get("inputTokens") is not None:
            metrics["prompt_tokens"] = float(usage.get("inputTokens", 0))
        if usage.get("outputTokens") is not None:
            metrics["completion_tokens"] = float(usage.get("outputTokens", 0))
        if usage.get("totalTokens") is not None:
            metrics["tokens"] = float(usage.get("totalTokens", 0))
        if usage.get("cacheReadInputTokens") is not None:
            metrics["prompt_cached_tokens"] = float(usage.get("cacheReadInputTokens", 0))
        if usage.get("cacheWriteInputTokens") is not None:
            metrics["completion_cached_tokens"] = float(usage.get("cacheWriteInputTokens", 0))

    return metrics


def converse_tracer(wrapped: Any, instance: Any, args: Any, kwargs: Any):
    """trace helper for `boto3.BedrockRuntime.Client.Converse` API"""

    input_data = _get_converse_input(**kwargs)
    metadata = _get_converse_request_metadata(**kwargs)

    with start_span(
        name="bedrock_runtime.converse",
        type=SpanTypeAttribute.LLM,
        input=input_data,
        metadata=metadata,
    ) as span:
        start_time = time.time()
        result: dict[str, Any] = wrapped(*args, **kwargs)
        end_time = time.time()

        output = parse_converse_result(result)
        metrics = _extract_converse_metrics(result, start_time, end_time)
        span.log(output=output, metrics=metrics)
        return result
