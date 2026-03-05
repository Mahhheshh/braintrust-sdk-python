from unittest.mock import MagicMock

import pytest
from braintrust.parameters import _extract_pydantic_fields, parameters_to_json_schema

DEFAULT_VALUE = "You are a helpful assistant."
DEFAULT_DESCRIPTION = "System prompt for the model"
SCHEMA_TITLE = "SystemPrompt"
PARAM_NAME = "system_prompt"

def test_extract_single_field_with_default_and_description():
    schema = {
        "value": {"type": "string", "default": DEFAULT_VALUE, "description": DEFAULT_DESCRIPTION},
    }
    defaults, descriptions = _extract_pydantic_fields(schema)
    assert defaults == DEFAULT_VALUE
    assert descriptions == DEFAULT_DESCRIPTION


def test_extract_single_field_missing_default_and_description():
    schema = {
        "value": {"type": "string"},
    }
    defaults, descriptions = _extract_pydantic_fields(schema)
    assert defaults is None
    assert descriptions is None


def test_extract_multi_field():
    schema = {
        "temperature": {"type": "number", "default": 0.7, "description": "Sampling temperature"},
        "max_tokens": {"type": "integer", "default": 1024, "description": "Maximum tokens to generate"},
    }
    defaults, descriptions = _extract_pydantic_fields(schema)
    assert defaults == {"temperature": 0.7, "max_tokens": 1024}
    assert descriptions == {"temperature": "Sampling temperature", "max_tokens": "Maximum tokens to generate"}


def test_extract_multi_field_partial_metadata():
    schema = {
        "temperature": {"type": "number", "default": 0.7},
        "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"},
    }
    defaults, descriptions = _extract_pydantic_fields(schema)
    assert defaults == {"temperature": 0.7, "max_tokens": None}
    assert descriptions == {"temperature": None, "max_tokens": "Maximum tokens to generate"}


def test_extract_empty_schema():
    defaults, descriptions = _extract_pydantic_fields({})
    assert defaults == {}
    assert descriptions == {}


@pytest.fixture
def v2_model():
    def _make(default=DEFAULT_VALUE, description=DEFAULT_DESCRIPTION):
        model = MagicMock()
        model.model_json_schema.return_value = {
            "title": SCHEMA_TITLE,
            "type": "object",
            "properties": {"value": {"type": "string", "default": default, "description": description}},
        }
        del model.get
        return model
    return _make


@pytest.fixture
def v1_model():
    def _make(default=DEFAULT_VALUE):
        model = MagicMock()
        del model.model_json_schema
        model.schema.return_value = {
            "title": SCHEMA_TITLE,
            "type": "object",
            "properties": {"value": {"type": "string", "default": default}},
        }
        del model.get
        return model
    return _make


@pytest.fixture
def v2_multi_field_model():
    def _make():
        model = MagicMock()
        model.model_json_schema.return_value = {
            "title": "ModelConfig",
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "default": 0.7, "description": "Sampling temperature"},
                "max_tokens": {"type": "integer", "default": 1024, "description": "Maximum tokens to generate"},
            },
        }
        del model.get
        return model
    return _make


@pytest.fixture
def v1_multi_field_model():
    def _make():
        model = MagicMock()
        del model.model_json_schema
        model.schema.return_value = {
            "title": "ModelConfig",
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "default": 0.7},
                "max_tokens": {"type": "integer", "default": 1024},
            },
        }
        del model.get
        return model
    return _make


def test_pydantic_v2_model(v2_model):
    schema = parameters_to_json_schema({PARAM_NAME: v2_model()})

    assert schema[PARAM_NAME]["type"] == "data"
    assert schema[PARAM_NAME]["default"] == DEFAULT_VALUE
    assert schema[PARAM_NAME]["description"] == DEFAULT_DESCRIPTION


def test_pydantic_v1_model(v1_model):
    schema = parameters_to_json_schema({PARAM_NAME: v1_model()})

    assert schema[PARAM_NAME]["type"] == "data"
    assert schema[PARAM_NAME]["default"] == DEFAULT_VALUE
    assert schema[PARAM_NAME]["description"] is None


def test_pydantic_v2_multi_field_model(v2_multi_field_model):
    schema = parameters_to_json_schema({PARAM_NAME: v2_multi_field_model()})

    assert schema[PARAM_NAME]["type"] == "data"
    assert schema[PARAM_NAME]["schema"]["title"] == "ModelConfig"
    assert schema[PARAM_NAME]["default"] == {"temperature": 0.7, "max_tokens": 1024}
    assert schema[PARAM_NAME]["description"] == {"temperature": "Sampling temperature", "max_tokens": "Maximum tokens to generate"}


def test_pydantic_v1_multi_field_model(v1_multi_field_model):
    schema = parameters_to_json_schema({PARAM_NAME: v1_multi_field_model()})

    assert schema[PARAM_NAME]["type"] == "data"
    assert schema[PARAM_NAME]["schema"]["title"] == "ModelConfig"
    assert schema[PARAM_NAME]["default"] == {"temperature": 0.7, "max_tokens": 1024}
    assert schema[PARAM_NAME]["description"] == {"temperature": None, "max_tokens": None}
