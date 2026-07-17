# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_baby_policy import BabyPolicyConfig
from .modeling_baby_policy import BabyPolicy
from .processor_baby_policy import make_baby_policy_pre_post_processors

__all__ = [
    "BabyPolicyConfig",
    "BabyPolicy",
    "make_baby_policy_pre_post_processors",
]
