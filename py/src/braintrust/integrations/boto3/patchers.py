from braintrust.integrations.base import NestedFunctionWrapperPatcher

from .tracing import converse_tracer


class Boto3ConversePatcher(NestedFunctionWrapperPatcher):
    """Patcher class for Boto3.BedrockRuntime.Client.Converse API"""

    name = "boto3"
    target_module = "botocore.client"
    target_path = "ClientCreator._create_client_class"
    target_attribute = "converse"
    nested_wrapper = converse_tracer
