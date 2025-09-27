# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
# from deepspeed import get_accelerator

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import torch._dynamo
torch._dynamo.config.verbose = False

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info
import numpy as np

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    # wandb.init(mode="disabled")

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            script_args=None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
    ):
        torch.set_float32_matmul_precision("high")
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # 필수: cache 비활성화

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # self.ref_model = None
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal_gen = script_args.temporal_gen

        self.quality_step = script_args.quality_step
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
            # use_cache=False,  # This incurs the Index error
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,  # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
            # use_cache=False, # This incurs the Index error
        )

        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,
            temperature=1,  # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            # use_cache=False, # This incurs the Index error
        )
        self.len_control = script_args.len_control
        self.beta = args.beta
        self.step_count = 0
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()
        logits = model(input_ids, **kwargs).logits.to(torch.float16)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def sample_gaussian_indices(self, T, num_points=2, mean_ratio=0.25, std_ratio=0.1):
        """
        T: total frames
        num_points: must be 2
        mean_ratio, std_ratio: Gaussian controls for the first point
        Returns two indices [start, end] such that end ≈ start + T // 2
        """
        assert num_points == 2, "This version only supports num_points == 2"

        segment_len = T // 2
        mean = T * mean_ratio
        std = T * std_ratio

        while True:
            start = int(np.random.normal(loc=mean, scale=std))
            end = start + segment_len

            # Ensure both indices are within valid range
            if 1 <= start < T - 1 and 1 < end < T:
                return sorted([start, end])

    def perturb_video_sequence_gaussian(self, video: torch.Tensor):
        """
        video: [T, C, H, W]
        Returns: perturbed_video: [T, C, H, W]
        Applies a single perturbation (repeat or reverse) to one segment.
        """
        T, C, H, W = video.shape

        T = video.size(0)
        # Sample 2 indices to define a single segment
        segment_points = self.sample_gaussian_indices(T, num_points=2)
        segment_points = sorted([0] + segment_points + [T])

        start, end = segment_points[1], segment_points[2]

        pre = video[:start]
        seg = video[start:end].clone()  # clone to avoid in-place modification
        post = video[end:]
        decision = random.random()
        if decision < 0.33:
            # Apply spatial "still" box instead of repeat
            h_box, w_box = H // 4, W // 4
            still_patch = video[start, :, :h_box, :w_box]  # [C, h_box, w_box]
            for t in range(seg.size(0)):
                seg[t, :, :h_box, :w_box] = still_patch
            new_video = torch.cat([pre, seg, post], dim=0)
        elif decision < 0.66:
            seg = torch.flip(seg, dims=[0])
            new_video = torch.cat([pre, seg, post], dim=0)
        else:
            new_video = torch.flip(video, dims=[0])

        if new_video.size(0) > T:
            new_video = new_video[:T]
        elif new_video.size(0) < T:
            pad = video[-1].unsqueeze(0).repeat(T - new_video.size(0), 1, 1, 1)
            new_video = torch.cat([new_video, pad], dim=0)

        return new_video

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 매 50 step마다 동기적 cache clear
        if self.step_count % 50 == 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        torch.cuda.synchronize()
        self.step_count += 1
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        input_copy = copy.deepcopy(inputs[0]['prompt'])

        input_copy = self.remove_none_from_data(input_copy)

        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = inputs[0]['path']
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = inputs[0]['path']
        print("compute_loss: ", inputs[0]['path'])

        # Check real video
        # Discriminator
        # if inputs[0]['solution'] == '<answer>B</answer>':
        # Quality
        # if inputs[0]['solution'] == '<answer>F</answer>':
        if inputs[0]['solution'] == '<answer>B</answer>':
            is_real_video = True
        else:
            is_real_video = False
        # For genvidbench: "ms" video already has 0-1 ranges
        if "ms/ms-" in inputs[0]['path']:
            do_rescale = False
        else:
            do_rescale = True


        image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)

        with torch.no_grad():
            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                do_rescale=do_rescale
            )

        """
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            # padding="max_length",
            truncation=True,
            add_special_tokens=False,
            # max_length=self.max_prompt_length,
        )
        """
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # min_len = min(prompt_inputs["input_ids"].shape[1], prompt_inputs["attention_mask"].shape[1])
        # prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -min_len:]
        # prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -min_len:]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        print(inputs[0]['path'], prompt_inputs["input_ids"].shape[1], prompt_inputs["attention_mask"].shape[1])

        if self.temporal_gen and video_inputs:
            perturbed_video = self.perturb_video_sequence_gaussian(video_inputs[0])
            target_device = prompt_inputs["input_ids"].device if "input_ids" in prompt_inputs else image_inputs[0].device
            perturbed_video = perturbed_video.to(dtype=video_inputs[0].dtype, device=target_device)

            assert perturbed_video.shape == video_inputs[0].shape

            gen_manipulated_video_inputs = [perturbed_video]
            # indices = torch.randperm(video_inputs[0].size(0))
            # shuffled_video_inputs = [video_inputs[0][indices]]
            # gen_manipulated_video_inputs = shuffled_video_inputs

            assert video_inputs[0].size() == perturbed_video.size(), (video_inputs[0].size(), perturbed_video.size())
            # 2:  torch.Size([12, 3, 252, 392])
            with torch.no_grad():
                gen_manipulated_prompt_inputs = self.processing_class(
                    text=copy.deepcopy(prompts_text),
                    images=image_inputs,
                    videos=gen_manipulated_video_inputs,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    do_rescale=do_rescale
                )
            gen_manipulated_prompt_inputs = super()._prepare_inputs(gen_manipulated_prompt_inputs)
            gen_manipulated_prompt_ids, gen_manipulated_prompt_mask = gen_manipulated_prompt_inputs["input_ids"], \
                                                                      gen_manipulated_prompt_inputs[
                                                                          "attention_mask"]

            if self.max_prompt_length is not None:
                gen_manipulated_prompt_ids = gen_manipulated_prompt_ids[:, -self.max_prompt_length:]
                gen_manipulated_prompt_mask = gen_manipulated_prompt_mask[:, -self.max_prompt_length:]
        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            with torch.no_grad():
                prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            if self.temporal_gen:
                if video_inputs:
                    with torch.no_grad():
                        gen_manipulated_prompt_completion_ids = unwrapped_model.generate(**gen_manipulated_prompt_inputs,
                                                                                         generation_config=self.shuffled_generation_config)

                    gen_manipulated_prompt_length = gen_manipulated_prompt_ids.size(1)

                    gen_manipulated_prompt_ids = gen_manipulated_prompt_completion_ids[:, :gen_manipulated_prompt_length]
                    gen_manipulated_completion_ids = gen_manipulated_prompt_completion_ids[:, gen_manipulated_prompt_length:]
                    gen_manipulated_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)

                    del perturbed_video


                else:
                    gen_manipulated_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs,
                                                                                     generation_config=self.dummy_generation_config)
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()

        if inputs[0]['data_type'] == 'video':
            # prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat_interleave(self.num_generations, dim=0)

            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)

            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1:]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)

        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids,
                                                                    **prompt_inputs).detach()

                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                # ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute the KL divergence between the model and the reference model

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1


        if self.temporal_gen and video_inputs:

            gen_manipulated_completions = self.processing_class.batch_decode(gen_manipulated_completion_ids,
                                                                             skip_special_tokens=True)
            if is_conversational(inputs[0]):
                gen_manipulated_completions = [[{"role": "assistant", "content": gen_manipulated_completion}] for
                                               gen_manipulated_completion
                                               in gen_manipulated_completions]

            # Compute the rewards
            gen_manipulated_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            gen_manipulated_rewards_per_func = torch.zeros(len(gen_manipulated_prompts), len(self.reward_funcs),
                                                           device=device)

            for i, (reward_func, reward_processing_class) in enumerate(
                    zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                gen_manipulated_reward_kwargs = {key: [] for key in inputs[0].keys() if
                                                 key not in ["prompt", "completion"]}

                for key in gen_manipulated_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        gen_manipulated_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)

                # Set the answer as "fake"
                # for quality
                # gen_manipulated_reward_kwargs["solution"] = ["<answer>C</answer>"] * self.shuffled_num_generations
                # for binary
                gen_manipulated_reward_kwargs["solution"] = ["<answer>A</answer>"] * self.shuffled_num_generations

                gen_manipulated_output_reward_func = reward_func(prompts=gen_manipulated_prompts,
                                                                 completions=gen_manipulated_completions,
                                                                 **gen_manipulated_reward_kwargs)
                # print("here: ", gen_manipulated_reward_kwargs.keys())
                # dict_keys(['problem_id', 'problem', 'data_type', 'problem_type', 'options', 'process', 'solution', 'path', 'data_source'])
                # gen_manipulated_reward_kwargs.keys():
                # accuracy_reward: 0
                # format_reward: 1
                gen_manipulated_rewards_per_func[:, i] = torch.tensor(gen_manipulated_output_reward_func,
                                                                      dtype=torch.float32,
                                                                      device=device)
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # Format: 1, temporal(fixed): 0.5, Acc: 1
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        if self.temporal_gen and video_inputs:
            gen_temporal_rewards_per_func = rewards_per_func.clone()

            # Case 1: 0/1 Discriminator
            # gen_manipulated_rewards_per_func[:, 0] = 1 - gen_manipulated_rewards_per_func[:, 0]

            # Case 2: A-E, F
            # if it is 0 reward: "solution is real" but answer is related to the fake
            # change to 1 if 0 else all 0

            # gen_manipulated_rewards_per_func:  tensor([0., 0., 1., 0.], device='cuda:0') tensor([0., 0., 1., 0.], device='cuda:0')
            #  tensor([1.0000, 3.2000, 3.2000, 1.0000, 3.2000, 3.0000, 3.2000, 1.0000],

            acc_mean = gen_temporal_rewards_per_func[:, 0].mean()
            gen_manipulated_acc_mean = gen_manipulated_rewards_per_func[:, 0].detach().mean()
            # For quality
            # if is_real_video and gen_manipulated_acc_mean >= 1.0:
            if is_real_video and acc_mean >= 0.75 and gen_manipulated_acc_mean == 1:
                mask = gen_temporal_rewards_per_func[:, 0] >= 1.0
                gen_temporal_rewards_per_func[mask, 0] += 0.2
                gen_temporal_rewards = (0.2 * mask.sum()).to('cuda')
            elif not is_real_video and acc_mean >= 0.75 and gen_manipulated_acc_mean == 1:
                mask = gen_temporal_rewards_per_func[:, 0] >= 1.0
                gen_temporal_rewards_per_func[mask, 0] += 0.1
                gen_temporal_rewards = (0.1 * mask.sum()).to('cuda')
            else:
                gen_temporal_rewards = torch.tensor(0.0).to('cuda')
        else:
            gen_temporal_rewards = torch.tensor(0.5).to('cuda')

        # Sum the rewards from all reward functions
        if self.temporal_gen and video_inputs:
            rewards = gen_temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)

        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            #             if len(selected_indices) > 1 and len(selected_indices) < self.num_generations:
            # if len(selected_indices) > 1:
            #     selected_items = [(i, lenth_list[i]) for i in selected_indices]
            #     sorted_items = sorted(selected_items, key=lambda x: x[1], reverse=True)
            #     N = len(sorted_items)
            #     for rank, (idx, length) in enumerate(sorted_items):
            #         reward = 0.2 - 0.2 * (rank / N)
            #         rewards[idx] += reward
            #         mem_rewards[idx] = reward
            # for idx in range(len(lenth_list)):
            #     if lenth_list[idx] >= 512:
            #         rewards[idx] -= 0.5

            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2

        # print(rewards)
        # print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # if self.len_control and len(selected_indices) == self.num_generations:
        #     for idx in range(len(rewards)):
        #         advantages[idx] += (mem_rewards[idx] - 0.2) * 2

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # import pdb
        # pdb.set_trace()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)

        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices

        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices

        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)


        if self.temporal_gen:
            gen_temporal_rewards_list = self.accelerator.gather_for_metrics(gen_temporal_rewards)
            self._metrics["gen_temporal_rewards"].append(
                self.accelerator.gather_for_metrics(gen_temporal_rewards_list).mean().item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        del video_inputs, prompt_inputs, completion_ids, ref_per_token_logps, per_token_logps
        torch.cuda.empty_cache()
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
