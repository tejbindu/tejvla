# processor_baby_policy.py
from typing import Any
import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_baby_policy_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = ...   # build your PolicyProcessorPipeline for inputs
    postprocessor = ...  # build your PolicyProcessorPipeline for outputs
    return preprocessor, postprocessor
