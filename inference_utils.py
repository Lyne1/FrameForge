# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

# Modified for FrameForgeM: final straight-through / reason-token / 3-expert release.

import argparse
import copy
import datetime
import json
import os

import loguru
import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state


def encode_vae(vae, images: torch.Tensor) -> torch.Tensor:
    if images.max() > 1.0 or images.min() < -1.0:
        raise ValueError(f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}")
    
    if images.ndim == 4:
        images = images.unsqueeze(2)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), vae.memory_efficient_context():
        latents = vae.encode(images).latent_dist.sample()
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            latents = latents * vae.config.scaling_factor
    
    return latents

def ref_video_encoder(image_path, vae, device, width=512, height=512):
    video_transforms = transforms.Compose(
        [
            transforms.Resize(min(width, height)),
            transforms.CenterCrop(min(width, height)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    image = Image.open(image_path)
    image = image.resize((height, width)).convert("RGB")
    
    frames = torch.from_numpy(np.array(image)).unsqueeze(0)
    # Normalize the reference image
    frames = frames.float()
    frames = frames.permute(0, 3, 1, 2).contiguous() # [F, C, H, W]
    frames = frames.float() / 255.0
    frames = video_transforms(frames)
    video = frames.to(device, dtype=vae.dtype).unsqueeze(0) # [B, F, C, H, W]
    video = video.permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W]
    vae_latents = encode_vae(vae, video)
    return vae_latents

def append_reasoning_latents(latents: torch.Tensor, num_reason_token: int) -> torch.Tensor:
    if num_reason_token < 1:
        raise ValueError("num_reason_token must be >= 1 when use_reasoning_latent is enabled.")

    reason_shape = (
        latents.shape[0],
        latents.shape[1],
        num_reason_token,
        latents.shape[3],
        latents.shape[4],
    )
    reason_latents = torch.randn(
        reason_shape,
        device=latents.device,
        dtype=latents.dtype,
    )
    return torch.cat([latents, reason_latents], dim=2)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task, transformer_version):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'transformer_version': transformer_version,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved generation config to: {config_path}")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  # When --flag is provided without value, enable it
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def load_checkpoint_to_transformer(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        model_state_dict = get_model_state_dict(pipe.transformer)
        dcp.load(
            state_dict={"model": model_state_dict},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading checkpoint: {e}", "ERROR")
        raise
