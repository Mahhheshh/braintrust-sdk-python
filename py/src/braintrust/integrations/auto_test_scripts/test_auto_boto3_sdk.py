import os

import boto3
from braintrust.auto import auto_instrument
from braintrust.integrations.boto3.patchers import Boto3ConversePatcher
from braintrust.integrations.test_utils import autoinstrument_test_context


def is_patched(target, patcher):
    return bool(getattr(target, patcher.nested_patch_marker_attr(), False))


# verify not patched initially
original_client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id="test-key",
    aws_secret_access_key="test-access-key",
    aws_session_token="test-session-key",
    region_name="us-east-1",
)

assert is_patched(original_client, Boto3ConversePatcher) is False

# instrument
results = auto_instrument()
assert results.get("boto3") is True

# patch
patched_client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_session_token="",
    region_name="us-east-1",
)

# idempotent
results = auto_instrument()
assert results.get("boto3") is True


assert is_patched(patched_client, Boto3ConversePatcher) is True

# make api call
with autoinstrument_test_context("test_auto_boto3", integration="boto3") as memory_logger:
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

    response = response = client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "what's 2+2"}]}],
        system=[{"text": "answer only in single integer"}],
    )

    spans = memory_logger.pop()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span["metadata"]["model_provider"] == "anthropic"
    assert "claude" in span["metadata"]["model"]
