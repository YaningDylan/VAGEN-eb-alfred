"""
SFT training for VAGEN-format EB-ALFRED data.

Self-contained trainer using Qwen2.5-VL with standard chat template.
No ERA special tokens (<|think_start|> etc.) - uses plain <think>/<answer> tags.

Usage:
    torchrun --nproc_per_node 2 scripts/train_sft_vagen.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path /root/workspace/era_sft_data/vagen_format/stage.yaml \
        --image_folder /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset \
        --output_dir /root/workspace/VAGEN-eb-alfred/exps/sft/checkpoints/qwen25vl3b-sft-vagen \
        --deepspeed scripts/ds_zero3.json
"""

import copy
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import yaml
from PIL import Image, ImageFile
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")

IGNORE_INDEX = -100


def rank0_print(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


# ── Arguments ──

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: Optional[str] = field(default=None)
    processor: Optional[object] = field(default=None)
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trajectory JSON for eval (base split only). If set, "
                  "n_eval_samples are drawn from base-split entries."},
    )
    n_eval_samples: int = field(
        default=20,
        metadata={"help": "Number of base-split samples to hold out for eval"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    freeze_visual_encoder: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)


# ── Dataset ──

class VagenSFTDataset(Dataset):
    """
    SFT dataset for VAGEN-format data.

    Expects JSON with list of samples, each having:
      - "image": path or list of paths (relative to image_folder)
      - "conversations": list of {from: system|human|gpt, value: str}

    The conversations use standard <think>/<answer> tags (no special tokens).
    """

    def __init__(self, tokenizer, processor, data_path, data_args, preloaded=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args
        self.list_data_dict = []
        self.list_image_path = []

        if preloaded is not None:
            self.list_data_dict = preloaded
            self.list_image_path = [""] * len(preloaded)
            rank0_print(f"Preloaded dataset: {len(self.list_data_dict)} samples")
            return

        if data_path.endswith(".yaml"):
            with open(data_path) as f:
                yaml_data = yaml.safe_load(f)
            for dataset in yaml_data.get("datasets", []):
                json_path = dataset["json_path"]
                images_folder = dataset.get("images_folder", "")
                sampling_strategy = dataset.get("sampling_strategy", "all")

                rank0_print(f"Loading {json_path} ({sampling_strategy})...")
                with open(json_path) as f:
                    cur_data = json.load(f)

                if ":" in sampling_strategy:
                    strategy, num = sampling_strategy.split(":")
                    num = int(num)
                    if strategy == "first":
                        cur_data = cur_data[:num]
                    elif strategy == "random":
                        random.shuffle(cur_data)
                        cur_data = cur_data[:num]

                rank0_print(f"  -> {len(cur_data)} samples")
                self.list_data_dict.extend(cur_data)
                self.list_image_path.extend([images_folder] * len(cur_data))
        else:
            with open(data_path) as f:
                cur_data = json.load(f)
            rank0_print(f"Loaded {len(cur_data)} samples from {data_path}")
            self.list_data_dict.extend(cur_data)
            self.list_image_path.extend([""] * len(cur_data))

        rank0_print(f"Total dataset size: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv.get("value", conv.get("content", "")).split())
                for conv in sample.get("conversations", [])
            )
            if "image" in sample:
                img = sample["image"]
                n_img = len(img) if isinstance(img, list) else (1 if img else 0)
                length_list.append(cur_len + 1200 * n_img)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i):
        try:
            return self._get_item(i)
        except Exception as e:
            rank0_print(f"Error fetching sample {i}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get_item(self, i):
        sample = self.list_data_dict[i]
        image_base = os.path.join(
            self.data_args.image_folder or "",
            self.list_image_path[i],
        )

        # Get image paths
        images = []
        if "image" in sample and sample["image"]:
            img_field = sample["image"]
            if isinstance(img_field, str):
                img_field = [img_field]
            images = [os.path.join(image_base, p) for p in img_field]

        # Build messages for chat template
        convs = sample["conversations"]
        roles_map = {"human": "user", "gpt": "assistant", "system": "system"}
        image_index = 0

        messages = []
        for conv in convs:
            role = roles_map.get(conv.get("from", conv.get("role", "")), "user")
            value = conv.get("value", conv.get("content", ""))

            content_parts = []
            # Count <image> placeholders
            n_images_in_text = value.count("<image>")
            if n_images_in_text > 0 and images:
                parts = value.split("<image>")
                for j, part in enumerate(parts):
                    if part.strip():
                        content_parts.append({"type": "text", "text": part})
                    if j < n_images_in_text and image_index < len(images):
                        content_parts.append({
                            "type": "image",
                            "image": images[image_index],
                        })
                        image_index += 1
            else:
                content_parts.append({"type": "text", "text": value})

            messages.append({"role": role, "content": content_parts})

        # Tokenize with chat template
        return self._tokenize_messages(messages, images)

    def _tokenize_messages(self, messages, image_paths):
        """Tokenize messages, masking user/system tokens."""
        tokenizer = self.tokenizer
        processor = self.processor

        # Build input_ids and labels turn by turn
        input_ids = []
        labels = []
        pixel_values = None
        image_grid_thw = None

        for msg in messages:
            role = msg["role"]
            single_msg = [msg]

            # Tokenize this single message
            text = tokenizer.apply_chat_template(
                single_msg,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Process images if present
            has_images = any(
                c.get("type") == "image" for c in msg.get("content", [])
            )

            if has_images:
                img_inputs, _ = process_vision_info(single_msg)
                inputs = processor(text=[text], images=img_inputs, return_tensors="pt")
                if pixel_values is None:
                    pixel_values = inputs["pixel_values"]
                    image_grid_thw = inputs["image_grid_thw"]
                else:
                    pixel_values = torch.cat([pixel_values, inputs["pixel_values"]], dim=0)
                    image_grid_thw = torch.cat([image_grid_thw, inputs["image_grid_thw"]], dim=0)
            else:
                inputs = processor(text=[text], return_tensors="pt")

            ids = inputs["input_ids"][0].tolist()
            input_ids.extend(ids)

            if role in ("user", "system"):
                labels.extend([IGNORE_INDEX] * len(ids))
            else:
                labels.extend(ids)

        # Add generation prompt at the end? No, training doesn't need it.

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        result = {"input_ids": input_ids, "labels": labels}
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw

        return result


# ── Data Collator ──

@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"][:self.tokenizer.model_max_length] for inst in instances]
        labels = [inst["labels"][:self.tokenizer.model_max_length] for inst in instances]

        pad_id = self.tokenizer.pad_token_id or 0

        # Pad
        max_len = max(len(ids) for ids in input_ids)
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.cat([torch.full((max_len - len(ids),), pad_id, dtype=torch.long), ids]) for ids in input_ids]
            labels = [torch.cat([torch.full((max_len - len(lb),), IGNORE_INDEX, dtype=torch.long), lb]) for lb in labels]
        else:
            input_ids = [torch.cat([ids, torch.full((max_len - len(ids),), pad_id, dtype=torch.long)]) for ids in input_ids]
            labels = [torch.cat([lb, torch.full((max_len - len(lb),), IGNORE_INDEX, dtype=torch.long)]) for lb in labels]

        batch = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(input_ids).ne(pad_id),
        }

        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([inst["pixel_values"] for inst in instances], dim=0)
            batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances], dim=0)

        return batch


# ── Training ──

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    # Load processor
    min_pixels = 500 * 500
    max_pixels = 500 * 500
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer = tokenizer
    data_args.processor = processor

    rank0_print(f"Processor: {processor.__class__.__name__}")

    # Freeze visual encoder (except merger)
    # In newer transformers, visual encoder is at model.model.visual
    visual_encoder = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if training_args.freeze_visual_encoder and visual_encoder is not None:
        for p in visual_encoder.parameters():
            p.requires_grad = False
        if hasattr(visual_encoder, "merger"):
            for p in visual_encoder.merger.parameters():
                p.requires_grad = True
        rank0_print("Visual encoder frozen (merger unfrozen)")

    # Build train dataset (all data)
    train_dataset = VagenSFTDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # Build eval dataset: base-split samples only, no overlap with train
    eval_dataset = None
    if data_args.eval_data_path and data_args.n_eval_samples > 0:
        with open(data_args.eval_data_path) as f:
            traj_data = json.load(f)
        # Filter to base split only (image path contains "/base/")
        base_samples = [s for s in traj_data if "/base/" in str(s.get("image", ""))]
        rng = random.Random(42)
        rng.shuffle(base_samples)
        eval_samples = base_samples[:data_args.n_eval_samples]
        rank0_print(f"Eval: {len(eval_samples)} base-split samples (from {len(base_samples)} total base)")

        eval_dataset = VagenSFTDataset(
            tokenizer=tokenizer,
            processor=processor,
            data_path=None,
            data_args=data_args,
            preloaded=eval_samples,
        )

    # Train
    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    rank0_print(f"Starting training with {len(train_dataset)} samples...")
    if list(os.scandir(training_args.output_dir)) if os.path.isdir(training_args.output_dir) else []:
        ckpts = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    else:
        trainer.train()

    trainer.save_state()

    # Save final model
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
