"""CPU inference checks; no dataset code, downloads or GPU allocation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

import infer
import inference_utils
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer as Transformer,
)


def tiny_transformer():
    return Transformer(
        hidden_size=24, heads_num=1, rope_dim_list=[8, 8, 8],
        mm_double_blocks_depth=1, mm_single_blocks_depth=1,
        text_states_dim=24, text_states_dim_2=24,
        text_projection="linear", attn_mode="torch",
    )


def inference_args(lora_path="unused.safetensors"):
    return infer.build_parser().parse_args([
        "--input_image", "unused.png", "--prompt", "test", "--model_path", "unused",
        "--output_path", "unused.png", "--dtype", "fp32", "--lora_path", str(lora_path),
    ])


class InferenceTests(unittest.TestCase):
    def test_hard_routing_ignores_reason_frame_and_padding(self):
        torch.manual_seed(12)
        model = tiny_transformer()
        model.set_moe_routing(3, "shared_seq")
        model.eval()
        image = torch.randn(2, 12, 24)
        text = torch.randn(2, 4, 24)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        with torch.no_grad():
            weights = model._compute_shared_gate_weights(image, text, mask, reason_index=1)
            self.assertTrue(torch.all((weights == 0) | (weights == 1)))
            torch.testing.assert_close(weights.sum(-1), torch.ones(2))
            image[:, 4:8] += 1000
            text[~mask.bool()] += 1000
            altered = model._compute_shared_gate_weights(image, text, mask, reason_index=1)
        torch.testing.assert_close(altered, weights)

    def test_three_expert_reasoning_rope(self):
        model = tiny_transformer()
        cos, sin = model.get_rotary_pos_embed_moe_reason((3, 2, 2), 3, 1)
        self.assertEqual(tuple(cos.shape), (3, 12, 24))
        self.assertEqual(cos.shape, sin.shape)
        self.assertTrue(torch.isfinite(cos).all())
        self.assertFalse(torch.equal(cos[0], cos[1]))

    def test_reasoning_latent_keeps_reference(self):
        reference = torch.randn(2, 4, 1, 8, 8)
        latents = inference_utils.append_reasoning_latents(reference, 1)
        self.assertEqual(tuple(latents.shape), (2, 4, 2, 8, 8))
        torch.testing.assert_close(latents[:, :, :1], reference)
        with self.assertRaises(ValueError):
            inference_utils.append_reasoning_latents(reference, 0)

    def test_final_inference_defaults(self):
        args = inference_args()
        self.assertTrue(args.shared_seq)
        self.assertTrue(args.use_reasoning_latent)
        self.assertEqual((args.num_experts, args.num_reason_token), (3, 1))
        self.assertEqual((args.lora_rank, args.lora_alpha), (32, 32))
        self.assertEqual(args.modules_to_save, "gate")
        self.assertTrue(args.disable_by5)
        self.assertFalse(args.sr)

    def test_loads_legacy_and_standard_lora_gate_weights(self):
        torch.manual_seed(42)
        base = tiny_transformer()
        fixture = copy.deepcopy(base)
        fixture.set_moe_routing(3, "shared_seq")
        fixture.add_adapter(LoraConfig(r=32, lora_alpha=32, target_modules="all-linear",
                                      modules_to_save=["gate"]))
        with torch.no_grad():
            for name, parameter in fixture.named_parameters():
                if "lora_" in name or "modules_to_save.default" in name:
                    parameter.normal_()
        expected = get_peft_model_state_dict(fixture)
        legacy = {key: value for key, value in fixture.state_dict().items()
                  if ".lora_" in key or ".modules_to_save.default." in key}
        for label, weights in (("standard", expected), ("legacy", legacy)):
            with self.subTest(format=label), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "adapter.safetensors"
                save_file(weights, str(checkpoint))
                pipe = SimpleNamespace(transformer=copy.deepcopy(base), device=torch.device("cpu"),
                                       apply_infer_optimization=Mock())
                with patch.object(inference_utils.HunyuanVideo_1_5_Pipeline, "create_pipeline", return_value=pipe) as factory:
                    loaded, _ = infer.build_pipeline(inference_args(checkpoint))
                self.assertTrue(factory.call_args.kwargs["disable_by5"])
                self.assertFalse(loaded.transformer.training)
                actual = get_peft_model_state_dict(loaded.transformer)
                for key, value in expected.items():
                    torch.testing.assert_close(actual[f"base_model.model.{key}"], value)


if __name__ == "__main__":
    unittest.main()
