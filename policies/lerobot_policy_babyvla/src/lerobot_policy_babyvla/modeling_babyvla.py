import torch
import torch.nn as nn
from typing import Any

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from .configuration_babyvla import BabyVLAConfig

class BabyVLAPolicy(PreTrainedPolicy):
    config_class = BabyVLAConfig  # must match the string in @register_subclass
    name = "babyvla"

    def __init__(self, config: BabyVLAConfig, dataset_stats: dict[str, Any] = None, dataset_meta: dict[str, Any] = None):
        super().__init__(config, dataset_stats, dataset_meta)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        self.model = ...  # your nn.Module here

    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        ...

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return {"params": self.parameters()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        ...

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called every step at inference)."""
        ...

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Compute the training loss.

        Returns `(loss, output_dict)`. `output_dict` may be `None`; everything in it must be
        logging-friendly Python natives (no tensors with gradients).

        `batch["action_is_pad"]` is a bool mask of shape (B, horizon) that marks
        timesteps padded because the episode ended before `horizon` steps; you
        can exclude those from your loss.
        """
        actions = batch[ACTION]
        action_is_pad = batch.get("action_is_pad")
        ...
        return loss, {"some_loss_component": some_loss_component.item()}
