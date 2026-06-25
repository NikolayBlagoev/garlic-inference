import torch
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time, sleep
# Qwen/Qwen3-0.6B
from power_profiler import PowerProfiler
MODEL_DIR = "qwen3-4b/"
# def encode(text: str):
#     return [1 + ord(c) - ord("a") for c in text]
# print(encode("representation"))
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# print(tokenizer.encode("1.42"))
# print(tokenizer.encode("1.4199999"))
# print(tokenizer.encode("hello"))
# exit()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="cuda")
model.eval()

sleep(10)
def run_prefill(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # print(f"Input: {text!r}  tokens={tokens}")
    input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    position_ids = torch.arange(len(tokens), dtype=torch.long, device="cuda").unsqueeze(0)
    next_token = input_ids
    past_key_values = None
    tm = None
    out = model(input_ids=next_token, past_key_values=past_key_values, position_ids=position_ids)
    
    for i in range(601):
            
            with torch.no_grad():

                out = model(input_ids=next_token, past_key_values=past_key_values, position_ids=position_ids)
                logits = out.logits[0]
                if i == 0:
                    tm = time()
                predicted = logits.argmax(dim=-1).tolist()
                # print(f"Argmax token IDs: {predicted}")
                print(tokenizer.decode(predicted),end="")
                past_key_values = out.past_key_values
                position_ids = torch.tensor([i + len(tokens)], dtype=torch.long, device="cuda").unsqueeze(0)
                next_token = torch.tensor([predicted[-1]], dtype=torch.long, device="cuda").unsqueeze(0)
    # print("Tok/s", 600/(time() - tm))

# run_prefill("Hello")
run_prefill("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")