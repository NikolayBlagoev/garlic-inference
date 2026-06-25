import torch
inp = torch.tensor([[0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9,0.23,0.1,0.109,0.0003,-0.4,0.25,0.9,9.7,0.109, 0.9]], dtype = torch.bfloat16)
router_probs = torch.nn.functional.softmax(inp, dtype=torch.float, dim=-1)

print(router_probs)
router_top_value, router_indices = torch.topk(router_probs, 4, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
print(router_top_value)
router_top_value = router_top_value.to(inp.dtype)
router_scores = router_top_value
print(router_top_value)
print(router_indices)
# class Qwen3MoeTopKRouter(nn.Module):
# //     def __init__(self, config):
# //         super().__init__()
# //         self.top_k = config.num_experts_per_tok
# //         self.num_experts = config.num_experts
# //         self.norm_topk_prob = config.norm_topk_prob
# //         self.hidden_dim = config.hidden_size
# //         self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

# //     def forward(self, hidden_states):
# //         hidden_states = hidden_states.reshape(-1, self.hidden_dim)
# //         router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
# //         router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
# //         router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
# //         if self.norm_topk_prob: -> when this is true removes the need for the previous softmax
# //             router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
# //         router_top_value = router_top_value.to(router_logits.dtype)
# //         router_scores = router_top_value
# //         return router_logits, router_scores, router_indices