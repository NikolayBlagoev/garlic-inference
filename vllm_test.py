import time
from openai import OpenAI
import pynvml
from power_profiler import PowerProfiler
# Connect to your local vLLM server
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
with PowerProfiler(handle):
    time.sleep(10)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

prompt = "Explain how KV-cache works in transformers."

with PowerProfiler(handle):
    response = client.chat.completions.create(
        model="qwen3-4b-fp8/",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )


generated_tokens = response.usage.completion_tokens
print("Generated tokens", generated_tokens)
# tps = generated_tokens / elapsed_time

# print(f"Generated {generated_tokens} tokens in {elapsed_time:.2f} seconds.")
# print(f"Speed: {tps:.2f} tokens/second")