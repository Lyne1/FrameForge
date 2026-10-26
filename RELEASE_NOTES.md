# Inference release notes

This inference-only export was prepared from the project's working tree,
including research changes, based on commit
`6d2448cad272cb9cd78c2068ba42a5d839bc3708`.

| Release file | Purpose |
| --- | --- |
| `infer.py` | Single-image inference, adapted from `generate_one_case.py` |
| `inference_utils.py` | Required helpers extracted from `generate_ip.py` |
| `hyvideo/` | Runtime model, pipeline, scheduler and utilities |
| `configs/final.sh` | Final inference settings |
| `scripts/infer.sh` | Portable single-GPU launcher |
| `tests/test_inference.py` | CPU inference checks |

The release retains shared-sequence routing, one reasoning latent frame,
three experts and LoRA rank/alpha 32/32. It loads gate parameters alongside
LoRA tensors and accepts the project's legacy adapter keys as well as standard
PEFT keys. Distributed initialization happens at execution time, so imports
and command-line help work without a GPU. The loaded transformer is put in
evaluation mode; reference encoding and reasoning noise use the inference seed.

The unused MetaQuery paths, custom adapter-saving method, trainers, optimizers,
dataset conversion code, training configuration and training tests are excluded.
Runtime model definitions retain their parameter structure for checkpoint
compatibility. Requirements contain inference dependencies only. Local paths,
weights, datasets, experiment outputs and Git history are not included.

Upstream license and third-party notices are retained. Validation covers CPU
inference tests, command-line imports, Python compilation and shell syntax;
full-model GPU generation has not been rerun for this export.
