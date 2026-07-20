import torch
import torch.nn as nn
from typing import Any

from lerobot.policies import PreTrainedPolicy
from .configuration_babyvla import BabyVLAConfig
import pickle
from .constants import ACTION, TASK, UP_IMAGE, SIDE_IMAGE, STATE, ACTION_PAD

class BabayModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
                nn.MaxPool2d(kernel_size=5),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5),
                nn.MaxPool2d(kernel_size=5),
                nn.Flatten(),
                nn.Linear(1296, 128),
                nn.Linear(128, 32),
                nn.Linear(32, 6),
                nn.Tanh()
                )

    def forward(self, batch: dict[str, torch.Tensor]):
        up_images = batch[UP_IMAGE] # (B, 3, 480, 640)
        return self.model(up_images)


class BabyVLAPolicy(PreTrainedPolicy):
    config_class = BabyVLAConfig  # must match the string in @register_subclass
    name = "babyvla"

    def __init__(self, config: BabyVLAConfig, dataset_stats: dict[str, Any] = None, dataset_meta: dict[str, Any] = None):
        super().__init__(config, dataset_stats, dataset_meta)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        self.model = BabayModel()
        self.criterion = nn.MSELoss()

    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        ...

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return self.model.parameters()

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

        actions = batch[ACTION] # (B, 50 (horizon), 6)
        #action_is_pad = batch.get(ACTION_PAD)
        #states = batch[STATE] # (B, 6)
        #up_images = batch[UP_IMAGE] # (B, 3, 480, 640)
        #side_images = batch[SIDE_IMAGE] # (B, 3, 480, 640)
        #loss = torch.tensor([1,2,3])
        imm_action = actions[:,0,:]
        pred_action = self.model(batch)
        loss = self.criterion(pred_action, imm_action)
        return loss, None


