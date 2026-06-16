from braintrust.integrations.base import BaseIntegration

from .patchers import Boto3ConversePatcher


class Boto3Integration(BaseIntegration):
    name = "boto3_integration"
    patchers = (Boto3ConversePatcher,)
    import_names = ("botocore",)
