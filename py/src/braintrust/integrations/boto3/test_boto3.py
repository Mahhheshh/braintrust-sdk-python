import boto3
import pytest
from braintrust import logger
from braintrust.integrations.test_utils import verify_autoinstrument_script
from braintrust.logger import Attachment
from braintrust.test_logger import init_test_logger

from .integration import Boto3Integration
from .tracing import _normalize_converse_message_content


PROJECT_NAME = "test-boto3"

Boto3Integration.setup()


@pytest.fixture
def bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id="",
        aws_secret_access_key="",
        aws_session_token="",
        region_name="us-east-1",
    )

    return bedrock_client


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def test_normalize_converse_message_content_converts_inline_media_bytes():
    normalized = _normalize_converse_message_content(
        [
            {"text": "summarize these"},
            {"image": {"format": "png", "source": {"bytes": b"image-data"}}},
            {
                "document": {
                    "format": "pdf",
                    "name": "report",
                    "source": {"bytes": b"pdf-data"},
                    "context": "annual report",
                    "citations": {"enabled": True},
                }
            },
            {"video": {"format": "mp4", "source": {"bytes": b"video-data"}}},
            {"audio": {"format": "mp3", "source": {"bytes": b"audio-data"}}},
        ]
    )

    assert normalized[0] == {"text": "summarize these"}
    image_attachment = normalized[1]["image"]["source"]["image_url"]["url"]
    assert isinstance(image_attachment, Attachment)
    assert image_attachment.reference["content_type"] == "image/png"

    document = normalized[2]["document"]
    assert document["context"] == "annual report"
    assert document["citations"] == {"enabled": True}
    document_attachment = document["source"]["file"]["file_data"]
    assert isinstance(document_attachment, Attachment)
    assert document_attachment.reference["content_type"] == "application/pdf"

    video_attachment = normalized[3]["video"]["source"]["file"]["file_data"]
    assert isinstance(video_attachment, Attachment)
    assert video_attachment.reference["content_type"] == "video/mp4"

    audio_attachment = normalized[4]["audio"]["source"]["file"]["file_data"]
    assert isinstance(audio_attachment, Attachment)
    assert audio_attachment.reference["content_type"] == "audio/mp3"


def test_normalize_converse_message_content_preserves_non_binary_sources_and_errors():
    normalized = _normalize_converse_message_content(
        [
            {
                "document": {
                    "format": "txt",
                    "name": "notes",
                    "source": {"text": "plain text", "content": [{"text": "chunk"}]},
                }
            },
            {
                "video": {
                    "format": "mp4",
                    "source": {"s3Location": {"uri": "s3://bucket/video.mp4", "bucketOwner": "123"}},
                }
            },
            {"audio": {"format": "mp3", "error": {"message": "could not decode"}}},
        ]
    )

    assert normalized[0]["document"]["source"] == {"text": "plain text", "content": [{"text": "chunk"}]}
    assert normalized[1]["video"]["source"] == {"s3Location": {"uri": "s3://bucket/video.mp4", "bucketOwner": "123"}}
    assert normalized[2]["audio"]["error"] == {"message": "could not decode"}


@pytest.mark.vcr()
def test_boto3_converse(bedrock_client, memory_logger):
    assert not memory_logger.pop()

    messages = [{"role": "user", "content": [{"text": "what's 2+2"}]}]
    model_provider = "anthropic"
    model_name = "claude-3-haiku-20240307-v1"

    response = bedrock_client.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[{"role": "user", "content": [{"text": "what's 2+2"}]}],
        system=[{"text": "answer only in single integer"}],
    )

    assert response["output"]["message"]["role"] == "assistant"
    assert "4" in response["output"]["message"]["content"][0]["text"]

    spans = memory_logger.pop()

    assert len(spans) == 1
    traced_span = spans[0]

    assert isinstance(traced_span["metrics"], dict)
    assert traced_span["span_attributes"]["name"] == "bedrock_runtime.converse"
    assert traced_span["span_attributes"]["type"] == "llm"

    assert traced_span["input"]["messages"] == messages

    assert traced_span["metadata"]["model"] == model_name
    assert traced_span["metadata"]["model_provider"] == model_provider

    output = response.get("output", None)
    print(traced_span)
    assert output is not None
    assert traced_span["output"] == output


class TestAutoInstrumentBoto3:
    def test_auto_instrument_anthropic(self):
        verify_autoinstrument_script("test_auto_boto3_sdk.py")
