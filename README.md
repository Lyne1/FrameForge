# FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation

[Project Page](https://lyne1.github.io/FrameForge-page/)

**Yijing Lin · Zhendong Mao · Mengqi Huang · Yuxin Wang · Yongdong Zhang**  
University of Science and Technology of China

FrameForge adapts the temporal priors of [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) for three conditional image-generation tasks with one model: **subject-driven generation**, **Canny-to-image generation**, and **instruction-guided image editing**. The condition image is treated as an earlier frame and the output image as a target frame. This repository provides the **inference code** for the three-expert, one-reasoning-latent configuration.

## Requirements and setup

Use Linux, Python 3.10, and a CUDA-capable GPU. Install a PyTorch build appropriate for your CUDA environment, then:

```bash
python -m pip install -r requirements.txt
```

### Checkpoints

Download the HunyuanVideo 1.5 base model and its text and vision encoders:

```bash
python -m pip install modelscope

hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
hf download black-forest-labs/FLUX.1-Redux-dev --local-dir ./ckpts/vision_encoder/siglip
```

The FLUX.1-Redux-dev download requires access to the gated Hugging Face repository. The resulting layout is:

```text
ckpts/
├── transformer/720p_t2v/
├── vae/
├── scheduler/
├── text_encoder/
│   ├── llm/
│   ├── byt5-small/
│   └── Glyph-SDXL-v2/
└── vision_encoder/siglip/
```

Download the [FrameForge LoRA checkpoint](https://huggingface.co/lyneeeeeeeee/FrameForge/blob/main/pytorch_lora_weights.safetensors):

```bash
hf download lyneeeeeeeee/FrameForge pytorch_lora_weights.safetensors --local-dir ./checkpoints
```

Use `./checkpoints/pytorch_lora_weights.safetensors` as `--lora_path`.

## Inference

Run the commands from the repository root. Replace the example paths with your own reference or condition image, prepared at **512 × 512 pixels**. All three tasks generate a single **512 × 512** output image and use 50 denoising steps.

The examples explicitly set `--reference_size 512`; inputs with other dimensions are resized to 512 × 512 before encoding. The launcher uses `--resolution 720p` to select the base model; image-generation output remains 512 × 512.

### Subject-driven generation

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=./ckpts bash scripts/infer.sh \
  --lora_path ./checkpoints/pytorch_lora_weights.safetensors \
  --input_image ./inputs/subject.jpg \
  --reference_size 512 \
  --prompt "the dog playing on a beach" \
  --output_path ./outputs/subject.png \
  --num_inference_steps 50
```

### Canny-to-image generation

Pass a Canny edge map as the input image. The manuscript uses guidance scale `2.0` for this task:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=./ckpts bash scripts/infer.sh \
  --lora_path ./checkpoints/pytorch_lora_weights.safetensors \
  --input_image ./inputs/edges.png \
  --reference_size 512 \
  --prompt "A red car parked beside a lake at sunset" \
  --output_path ./outputs/canny.png \
  --num_inference_steps 50 --guidance_scale 2.0
```

### Instruction-guided image editing

Pass the source image and an editing instruction. The default guidance scale is `6.0`:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=./ckpts bash scripts/infer.sh \
  --lora_path ./checkpoints/pytorch_lora_weights.safetensors \
  --input_image ./inputs/edit.png \
  --reference_size 512 \
  --prompt "Change the sky to sunset while preserving the rest of the scene" \
  --output_path ./outputs/edited.png \
  --num_inference_steps 50
```

The launcher uses one GPU and reads the final configuration from [configs/final.sh](configs/final.sh).


## Citation

If you use FrameForge in research, please cite the manuscript. The entry below is provisional and can be updated when publication metadata is available.

```bibtex
@misc{lin2026frameforge,
  title  = {FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation},
  author = {Yijing Lin and Zhendong Mao and Mengqi Huang and Yuxin Wang and Yongdong Zhang},
  year   = {2026}
}
```

## License and validation

This project is derived from HunyuanVideo 1.5 and retains its [Tencent Hunyuan Community License](LICENSE) and [third-party notices](NOTICE). Please review those terms before redistributing code or weights.
