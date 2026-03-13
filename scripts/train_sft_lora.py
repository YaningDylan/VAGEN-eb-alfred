"""
SFT training with LoRA for VAGEN-format EB-ALFRED data.

Uses PEFT LoRA on Qwen2.5-VL-3B-Instruct. Much faster and less memory than full-param.

Usage:
    python scripts/train_sft_lora.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path /root/workspace/era_sft_data/vagen_format/stage.yaml \
        --image_folder /root/workspace/era_sft_data/EB-ALFRED_trajectory_augmented_prior_dataset \
        --output_dir /root/workspace/VAGEN-eb-alfred/exps/sft/checkpoints/qwen25vl3b-lora
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
import yaml
from PIL import ImageFile
from peft import LoraConfig, get_peft_model, TaskType
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
        print(*args, **kwargs, flush=True)


# ── Arguments ──

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: Optional[str] = field(default=None)
    processor: Optional[object] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096)
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")


# ── Dataset ──

class VagenSFTDataset(Dataset):
    def __init__(self, tokenizer, processor, data_path, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args
        self.list_data_dict = []
        self.list_image_path = []

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

        images = []
        if "image" in sample and sample["image"]:
            img_field = sample["image"]
            if isinstance(img_field, str):
                img_field = [img_field]
            images = [os.path.join(image_base, p) for p in img_field]

        convs = sample["conversations"]
        roles_map = {"human": "user", "gpt": "assistant", "system": "system"}
        image_index = 0

        messages = []
        for conv in convs:
            role = roles_map.get(conv.get("from", conv.get("role", "")), "user")
            value = conv.get("value", conv.get("content", ""))

            content_parts = []
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

        return self._tokenize_messages(messages)

    def _tokenize_messages(self, messages):
        """Tokenize all messages at once, then mask non-assistant tokens."""
        processor = self.processor

        # Process images first to know actual sizes
        all_images, _ = process_vision_info(messages)

        # Use processor.apply_chat_template so image token counts match features
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        if all_images:
            inputs = processor(text=[full_text], images=all_images, return_tensors="pt")
        else:
            inputs = processor(text=[full_text], return_tensors="pt")

        input_ids = inputs["input_ids"][0]

        # Create labels: mask everything except assistant responses
        labels = input_ids.clone()

        # Find assistant response boundaries using special tokens
        # Qwen2.5-VL chat template uses:
        #   <|im_start|>assistant\n ... <|im_end|>
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Tokenize "assistant\n" to find the marker
        assistant_marker = self.tokenizer.encode("assistant\n", add_special_tokens=False)

        ids_list = input_ids.tolist()
        in_assistant = False
        i = 0
        while i < len(ids_list):
            if ids_list[i] == im_start_id:
                # Check if next tokens are "assistant\n"
                marker_end = i + 1 + len(assistant_marker)
                if marker_end <= len(ids_list) and ids_list[i+1:marker_end] == assistant_marker:
                    # Mask the <|im_start|>assistant\n part
                    for j in range(i, marker_end):
                        labels[j] = IGNORE_INDEX
                    i = marker_end
                    in_assistant = True
                    continue
                else:
                    in_assistant = False
            elif ids_list[i] == im_end_id:
                if in_assistant:
                    # Keep the <|im_end|> token in labels (train to predict end)
                    in_assistant = False
                    i += 1
                    continue

            if not in_assistant:
                labels[i] = IGNORE_INDEX
            i += 1

        # Truncate to model_max_length
        max_len = self.tokenizer.model_max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]

        result = {"input_ids": input_ids, "labels": labels}
        if "pixel_values" in inputs:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            image_grid_thw = inputs["image_grid_thw"]
            pixel_values = inputs["pixel_values"]

            # Count how many image_pad tokens remain after truncation
            n_tokens_remaining = (input_ids == image_token_id).sum().item()

            if n_tokens_remaining == 0:
                # All images were truncated, don't include pixel_values
                pass
            else:
                # Figure out how many complete images fit
                cumulative = 0
                keep_images = 0
                for idx in range(len(image_grid_thw)):
                    t, h, w = image_grid_thw[idx].tolist()
                    img_tokens = int(t * h * w)
                    if cumulative + img_tokens <= n_tokens_remaining:
                        cumulative += img_tokens
                        keep_images += 1
                    else:
                        break

                if keep_images > 0 and cumulative == n_tokens_remaining:
                    # Keep only the images that fully fit
                    # Compute pixel count per image from grid_thw
                    # Each grid cell = merge_size^2 * temporal_patch_size patches = 4 patches
                    # Each patch = patch_size^2 pixels in the pixel_values tensor
                    pixel_counts = []
                    for idx in range(keep_images):
                        t, h, w = image_grid_thw[idx].tolist()
                        # pixel_values has shape (total_patches, channel_dim)
                        # patches per image = t * h * w * merge_size^2 = t * h * w * 4
                        pixel_counts.append(int(t * h * w * 4))

                    total_pixels = sum(pixel_counts)
                    result["pixel_values"] = pixel_values[:total_pixels]
                    result["image_grid_thw"] = image_grid_thw[:keep_images]
                else:
                    # Partial image at boundary - remove all remaining image tokens
                    input_ids[input_ids == image_token_id] = self.tokenizer.pad_token_id or 0
                    result["input_ids"] = input_ids

        return result


# ── Data Collator ──

@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]

        pad_id = self.tokenizer.pad_token_id or 0

        longest = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        for ids, lb in zip(input_ids, labels):
            pad_len = longest - len(ids)
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([lb, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
            )

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_input_ids).ne(pad_id),
        }

        # Handle pixel_values (some samples may not have images)
        has_pv = [("pixel_values" in inst) for inst in instances]
        if any(has_pv):
            pvs = [inst["pixel_values"] for inst in instances if "pixel_values" in inst]
            thws = [inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst]
            batch["pixel_values"] = torch.cat(pvs, dim=0)
            batch["image_grid_thw"] = torch.cat(thws, dim=0)

        return batch


# ── Training ──

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False

    # Freeze visual encoder entirely
    visual_encoder = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual_encoder is not None:
        for p in visual_encoder.parameters():
            p.requires_grad = False
        rank0_print("Visual encoder frozen")

    # Apply LoRA
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=500 * 500,
        max_pixels=500 * 500,
    )
    processor.tokenizer = tokenizer
    data_args.processor = processor

    # Build dataset
    train_dataset = VagenSFTDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    rank0_print(f"Starting LoRA training with {len(train_dataset)} samples...")
    if os.path.isdir(training_args.output_dir):
        ckpts = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    else:
        trainer.train()

    # Save LoRA adapter
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    rank0_print(f"LoRA adapter saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
