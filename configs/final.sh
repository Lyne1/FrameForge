#!/usr/bin/env bash
# FrameForge inference: shared-sequence hard top-1 routing.
# Matches the straight-through / reason-token / 3-expert checkpoint.
NUM_EXPERTS=3
MOE_ROUTING_MODE=shared_seq
USE_REASONING_LATENT=true
NUM_REASON_TOKEN=1
LORA_RANK=32
LORA_ALPHA=32
