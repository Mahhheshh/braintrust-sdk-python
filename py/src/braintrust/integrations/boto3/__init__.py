import logging

from braintrust.logger import NOOP_SPAN, current_span, init_logger

from .integration import Boto3Integration


logger = logging.getLogger(__name__)

__all__ = ["Boto3Integration", "setup_boto3"]


def setup_boto3(
    api_key: str | None = None,
    project_id: str | None = None,
    project_name: str | None = None,
) -> bool:
    span = current_span()

    if span == NOOP_SPAN:
        init_logger(project=project_name, api_key=api_key, project_id=project_id)

    return Boto3Integration.setup()
