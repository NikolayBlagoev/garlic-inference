"""
Reference implementation debug script for Qwen3-30B-A3B-FP8.

Monkey patches Qwen3MoeSparseMoeBlock and Qwen3MoeTopKRouter to print
inputs/outputs at each forward call so our custom C++ kernel outputs
can be compared against the HuggingFace reference.

Apply patches BEFORE from_pretrained so they're in place when the model
runs (Python looks up class methods at call time, so patching after load
also works, but patching before is cleaner).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

MODEL_PATH = "qwen3-30b-fp8/"


# ── stat helper ───────────────────────────────────────────────────────────────

def _stat(t: torch.Tensor, label: str = "") -> str:
    f = t.detach().float()
    prefix = f"{label} " if label else ""
    return (
        f"{prefix}shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min().item():.5f} max={f.max().item():.5f} "
        f"mean={f.mean().item():.5f} std={f.std().item():.5f}"
    )


# ── router patch ──────────────────────────────────────────────────────────────

_router_call_count = 0
_orig_router_forward = Qwen3MoeTopKRouter.forward


def _patched_router_forward(self, hidden_states: torch.Tensor):
    global _router_call_count
    idx = _router_call_count
    _router_call_count += 1

    router_logits, router_scores, router_indices = _orig_router_forward(self, hidden_states)

    n_show = min(4, router_indices.shape[0])
    print(f"\n{'='*60}")
    print(f"[Router #{idx}]")
    print(f"  {_stat(hidden_states, 'input hidden_states')}")
    print(f"  {_stat(router_logits, 'router_logits (post-softmax)')}")
    print(f"  {_stat(router_scores, 'top-k weights')}")
    print(f"  selected experts (first {n_show} tokens): "
          f"{router_indices[:n_show].tolist()}")
    print(f"  top-k weights   (first {n_show} tokens): "
          f"{router_scores[:n_show].float().tolist()}")

    return router_logits, router_scores, router_indices


Qwen3MoeTopKRouter.forward = _patched_router_forward


# ── sparse-moe-block patch ────────────────────────────────────────────────────

_moe_call_count = 0
_orig_moe_forward = Qwen3MoeSparseMoeBlock.forward


def _patched_moe_forward(self, hidden_states: torch.Tensor):
    global _moe_call_count
    idx = _moe_call_count
    _moe_call_count += 1

    print(f"\n{'='*60}")
    print(f"[SparseMoeBlock #{idx}]")
    print(f"  {_stat(hidden_states, 'input')}")

    output = _orig_moe_forward(self, hidden_states)

    print(f"  {_stat(output, 'output')}")
    n_show = min(8, output.reshape(-1, output.shape[-1]).shape[0])
    print(f"  output first {n_show} token vecs [0..4]: "
          f"{output.reshape(-1, output.shape[-1])[:n_show, :5].float().tolist()}")

    return output


Qwen3MoeSparseMoeBlock.forward = _patched_moe_forward


# ── load model ────────────────────────────────────────────────────────────────

print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (this may take a while for 30B) …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",   # respects the FP8 quantization config in config.json
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}")


# ── run a single forward pass ─────────────────────────────────────────────────

PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

# text = tokenizer.encode(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

print(f"\nRunning forward pass. Input tokens: {inputs['input_ids'].shape}\n")
with torch.no_grad():
    outputs = model(**inputs)

print(f"\n{'='*60}")
print(f"Done. logits shape: {outputs.logits.shape}")
print(f"Total SparseMoeBlock calls: {_moe_call_count}")
print(f"Total Router calls:         {_router_call_count}")

# Show the top-5 predicted next tokens
top5 = outputs.logits[0, -1].topk(5)
print("\nTop-5 next tokens:")
for score, tok_id in zip(top5.values.float().tolist(), top5.indices.tolist()):
    print(f"  {tok_id:6d}  {tokenizer.decode([tok_id])!r:15s}  logit={score:.4f}")
