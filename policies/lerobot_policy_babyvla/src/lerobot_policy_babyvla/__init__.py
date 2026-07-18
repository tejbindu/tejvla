# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_babyvla import BabyVLAConfig
from .modeling_babyvla import BabyVLAPolicy
from .processor_babyvla import make_babyvla_pre_post_processors

__all__ = [
    "BabyVLAConfig",
    "BabyVLAPolicy",
    "make_babyvla_pre_post_processors",
]
