import torch
import torch.nn as nn
from typing import Any

from lerobot.policies import PreTrainedPolicy
from .configuration_babyvla import BabyVLAConfig
import pickle
from .constants import ACTION, TASK, UP_IMAGE, SIDE_IMAGE, STATE, ACTION_PAD
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = list(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask


class BabyModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", device_map="auto", quantization_config=self.quantization_config)
        self.processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B")
        self.action_tokens = [
                f"<action_{i}>" for i in range(100)
                ]
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.action_tokens
            })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.allowed_ids = [
                self.tokenizer.convert_tokens_to_ids(token) for token in self.action_tokens
                ] + [self.tokenizer.eos_token_id]
        self.buckets = torch.tensor([-1 + i/50 for i in range(101)]).to("cuda")
        assert len(self.allowed_ids) == len(self.buckets)

    def forward(self, batch: dict[str, torch.Tensor]):
        up_images = batch[UP_IMAGE] # (B, 3, 480, 640)
        side_images = batch[SIDE_IMAGE] # (B, 3, 480, 640)
        cat_images = torch.cat((up_images, side_images), dim=-1)
        tasks = batch[TASK] # (B)
        prompts = [f"The task given to robot is: {tasks[i]} .This is the robot view (up view is on the left and side_view is on the right) <|image_pad|>. Robot should take action " for i in range(len(tasks))]
        inputs = self.processor(images=cat_images, text=prompts, return_tensors="pt").to(self.model.device)
        #
        actions = batch[ACTION]
        imm_action = actions[:,0,:].to("cuda")
        imm_action = torch.bucketize(imm_action, self.buckets)-1
        action_ids = torch.tensor(self.allowed_ids).to("cuda")[imm_action]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], action_ids], dim=1,)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(action_ids),],dim=1,)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat(
                    [
                        inputs["mm_token_type_ids"],torch.zeros_like(action_ids),],dim=1,)
        labels = torch.full_like(inputs["input_ids"], -100)
        labels[:, -6:] = action_ids
        outputs = self.model(**inputs,labels=labels)
        return outputs
        #
        # outputs = self.model.generate(**inputs, min_new_tokens=6, max_new_tokens=6, logits_processor=LogitsProcessorList([AllowedTokensLogitsProcessor(self.allowed_ids)]))
        # return outputs[:,-6:]


class BabyVLAPolicy(PreTrainedPolicy):
    config_class = BabyVLAConfig  # must match the string in @register_subclass
    name = "babyvla"

    def __init__(self, config: BabyVLAConfig, dataset_stats: dict[str, Any] = None, dataset_meta: dict[str, Any] = None):
        super().__init__(config, dataset_stats, dataset_meta)
        config.validate_features()  # not called automatically by the base class
        self.config = config
        self.model = BabyModel().to("cuda")
        self.criterion = nn.CrossEntropyLoss().to("cuda")

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
        # with open("batch.pkl", "wb") as f:
        #     pickle.dump(batch, f)
        actions = batch[ACTION] # (B, 50 (horizon), 6)
        tasks = batch[TASK] # (B)
        #action_is_pad = batch.get(ACTION_PAD)
        #states = batch[STATE] # (B, 6)
        #up_images = batch[UP_IMAGE] # (B, 3, 480, 640)
        #side_images = batch[SIDE_IMAGE] # (B, 3, 480, 640)
        #loss = torch.tensor([1,2,3])
        imm_action = actions[:,0,:].to("cuda")
        output = self.model(batch)
        # print("foo")
        # print(pred_action)
        # print("moo")
        # loss = self.criterion(pred_action, imm_action)
        return output.loss, None


