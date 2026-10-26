# Modified for FrameForgeM: final straight-through / reason-token / 3-expert release.
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE

"""Generate one image from one reference image and one text prompt.

This entry point shares the model setup and image-latent helpers from
``inference_utils.py``, but does not depend on DreamBench directory layouts,
numeric instance IDs, or predefined prompt dictionaries.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

import inference_utils as ip


def build_pipeline(args):
    """Create and configure the pipeline for the FrameForgeM final recipe."""
    infer_state = ip.initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError(
            "sparse_attn and use_sageattn cannot be enabled simultaneously."
        )

    if args.use_fp8_gemm and "sgl" in args.quant_type:
        try:
            import sgl_kernel  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "sgl_kernel is required for the selected quantization type. "
                "Install it with `pip install sgl-kernel==0.3.18`."
            ) from exc

    if args.enable_step_distill and args.enable_cache:
        raise ValueError(
            "Enabling both step distillation and cache degrades performance."
        )

    task = "t2v"
    transformer_version = ip.HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution,
        task,
        args.cfg_distilled,
        args.enable_step_distill,
        args.sparse_attn,
    )

    if args.dtype == "bf16":
        transformer_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    enable_offloading = args.offloading
    if args.group_offloading is None:
        offloading_config = (
            ip.HunyuanVideo_1_5_Pipeline.get_offloading_config()
        )
        enable_group_offloading = offloading_config["enable_group_offloading"]
    else:
        enable_group_offloading = args.group_offloading

    device = torch.device("cpu" if enable_offloading else "cuda")
    transformer_init_device = (
        torch.device("cpu") if enable_group_offloading else device
    )

    pipe = ip.HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=args.sr,
        disable_by5=args.disable_by5,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
        flow_shift=args.flow_shift,
    )

    if args.shared_seq and args.set_moe:
        raise ValueError(
            "shared_seq and set_moe cannot be enabled at the same time."
        )

    if args.shared_seq:
        pipe.transformer.set_moe_routing(
            num_experts=args.num_experts,
            routing_mode="shared_seq",
        )
        pipe.transformer.to(pipe.device, dtype=transformer_dtype)

    if args.set_moe:
        pipe.transformer.set_moe(args.num_experts)
        pipe.transformer.to(pipe.device, dtype=transformer_dtype)

    ip.loguru.logger.info(
        f"{enable_offloading=} {enable_group_offloading=} "
        f"overlap_group_offloading={args.overlap_group_offloading}"
    )
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=args.overlap_group_offloading,
    )

    if args.checkpoint_path:
        ip.load_checkpoint_to_transformer(pipe, args.checkpoint_path)

    if args.lora_path:
        lora_kwargs = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_name,
        }
        if args.modules_to_save:
            lora_kwargs["modules_to_save"] = args.modules_to_save.split(",")

        pipe.transformer = ip.get_peft_model(
            pipe.transformer,
            ip.LoraConfig(**lora_kwargs),
        )
        raw_state_dict = ip.load_file(args.lora_path)
        # The project's FSDP-aware saver retains adapter names; standard PEFT
        # files omit them. Normalize both formats before PEFT restores wrappers.
        lora_state_dict = {}
        for key, value in raw_state_dict.items():
            key = key.replace(".lora_A.default.", ".lora_A.")
            key = key.replace(".lora_B.default.", ".lora_B.")
            key = key.replace(".modules_to_save.default.", ".")
            if not key.startswith("base_model.model."):
                key = f"base_model.model.{key}"
            lora_state_dict[key] = value
        try:
            result = ip.set_peft_model_state_dict(pipe.transformer, lora_state_dict, adapter_name="default")
        except KeyError as exc:
            raise ValueError(f"Checkpoint is missing a required adapter/gate tensor: {exc}") from exc
        missing_adapter = [key for key in result.missing_keys
                           if ".lora_" in key or ".modules_to_save.default." in key]
        if result.unexpected_keys or missing_adapter:
            raise ValueError(
                "LoRA/gate checkpoint does not match the selected configuration: "
                f"missing={missing_adapter}, unexpected={result.unexpected_keys}"
            )
        ip.rank0_log("Loaded LoRA and gate weights.", "INFO")
    pipe.transformer.eval()

    if args.sr and hasattr(pipe, "sr_pipeline"):
        sr_infer_state = ip.copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=args.overlap_group_offloading,
        )

    return pipe, transformer_version


def resolve_output_path(args) -> Path:
    """Treat output_path as a filename when it has an image suffix, else a dir."""
    output_path = Path(args.output_path).expanduser()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    if output_path.suffix.lower() in image_suffixes:
        return output_path

    output_name = args.output_name or f"{Path(args.input_image).stem}.png"
    if Path(output_name).suffix.lower() not in image_suffixes:
        output_name = f"{output_name}.png"
    return output_path / output_name


def generate_one_case(args):
    output_path = resolve_output_path(args)
    pipe, transformer_version = build_pipeline(args)

    if args.video_length != 121:
        ip.rank0_log(
            "121 frames is optimal for video generation; "
            f"generating {args.video_length} frames for this image case.",
            "WARNING",
        )
    if not args.rewrite:
        ip.rank0_log(
            "Prompt rewriting is disabled; this may affect output quality.",
            "WARNING",
        )

    ref_latents = ip.ref_video_encoder(
        args.input_image,
        pipe.vae,
        "cuda",
        args.reference_size,
        args.reference_size,
    )
    if args.use_reasoning_latent:
        ref_latents = ip.append_reasoning_latents(
            ref_latents,
            args.num_reason_token,
        )

    pipeline_kwargs = {}
    if args.guidance_scale is not None:
        pipeline_kwargs["guidance_scale"] = args.guidance_scale

    with torch.no_grad():
        sample = pipe(
            enable_sr=args.sr,
            prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.video_length,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            output_type="pt",
            prompt_rewrite=args.rewrite,
            return_pre_sr_video=args.save_pre_sr_video,
            generate_image=True,
            ref_latents=ref_latents,
            **pipeline_kwargs,
        ).videos[0]

    if int(os.environ.get("RANK", "0")) == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = torch.from_numpy(np.array(sample.squeeze(1)))
        ip.save_image(image, str(output_path))
        ip.rank0_log(f"Saved generated image to: {output_path}", "INFO")
        if args.save_generation_config:
            ip.save_config(
                args,
                str(output_path),
                "one_case_image_generation",
                transformer_version,
            )


def add_bool_argument(parser, name, default, help_text):
    parser.add_argument(
        f"--{name}",
        type=ip.str_to_bool,
        nargs="?",
        const=True,
        default=default,
        help=help_text,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate one image from one input image and one prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_image",
        required=True,
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt/instruction for image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        default="",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output image path, or a directory in which to save the image.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Filename used when output_path is a directory.",
    )
    parser.add_argument(
        "--resolution",
        choices=["480p", "720p"],
        default="720p",
    )
    parser.add_argument("--aspect_ratio", default="16:9")
    parser.add_argument("--reference_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--video_length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
    )

    add_bool_argument(parser, "sr", False, "Enable super resolution.")
    add_bool_argument(
        parser,
        "save_pre_sr_video",
        False,
        "Return the pre-super-resolution result.",
    )
    add_bool_argument(parser, "rewrite", False, "Enable prompt rewriting.")
    add_bool_argument(
        parser,
        "cfg_distilled",
        False,
        "Use the CFG-distilled model.",
    )
    add_bool_argument(
        parser,
        "enable_step_distill",
        False,
        "Enable step-distilled inference.",
    )
    add_bool_argument(parser, "sparse_attn", False, "Enable sparse attention.")
    add_bool_argument(parser, "offloading", False, "Enable CPU offloading.")
    add_bool_argument(
        parser,
        "group_offloading",
        False,
        "Enable group offloading (disabled in the final recipe).",
    )
    add_bool_argument(
        parser,
        "overlap_group_offloading",
        True,
        "Overlap group offloading.",
    )
    add_bool_argument(parser, "use_sageattn", False, "Enable SageAttention.")
    add_bool_argument(
        parser,
        "enable_torch_compile",
        False,
        "Enable torch.compile.",
    )
    add_bool_argument(parser, "enable_cache", False, "Enable feature cache.")
    add_bool_argument(parser, "use_fp8_gemm", False, "Enable FP8 GEMM.")
    add_bool_argument(parser, "set_moe", False, "Enable standard MoE.")
    add_bool_argument(
        parser,
        "shared_seq",
        True,
        "Enable shared-sequence MoE routing.",
    )
    add_bool_argument(
        parser,
        "use_reasoning_latent",
        True,
        "Append reasoning latent frames.",
    )
    add_bool_argument(
        parser,
        "save_generation_config",
        False,
        "Save a JSON file containing generation parameters.",
    )

    parser.add_argument("--sage_blocks_range", default="0-53")
    parser.add_argument("--cache_type", default="deepcache")
    parser.add_argument("--no_cache_block_id", default="53")
    parser.add_argument("--cache_start_step", type=int, default=11)
    parser.add_argument("--cache_end_step", type=int, default=45)
    parser.add_argument("--total_steps", type=int, default=50)
    parser.add_argument("--cache_step_interval", type=int, default=4)
    add_bool_argument(parser, "disable_by5", True, "Disable byT5, matching the final checkpoint configuration.")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--lora_path", required=True, help="Trained LoRA safetensors file including the saved gate.")
    parser.add_argument("--quant_type", default="fp8-per-token-sgl")
    parser.add_argument("--include_patterns", default="double_blocks")
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--target_name",
        default="all-linear",
    )
    parser.add_argument("--modules_to_save", default="gate")
    parser.add_argument("--num_reason_token", type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        parser.error(f"Input image not found: {args.input_image}")
    if not args.prompt.strip():
        parser.error("--prompt cannot be empty")
    if args.reference_size <= 0:
        parser.error("--reference_size must be greater than zero")
    if args.use_reasoning_latent and args.num_reason_token < 1:
        parser.error(
            "--num_reason_token must be >= 1 when reasoning latent is enabled"
        )

    if not os.path.isfile(args.lora_path):
        parser.error(f"LoRA file not found: {args.lora_path}")
    if not torch.cuda.is_available():
        parser.error("Inference requires a CUDA GPU")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    try:
        ip.initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        generate_one_case(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
