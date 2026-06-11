import pynvml
import time
class PowerProfiler:
    def __init__(self, handle):
        self.power_before_gpu = 0
        self.power_after_gpu = 0
        self.power_before_cpu = 0
        self.power_after_cpu = 0
        self.tm_start = 0
        self.tm_end = 0
        self.handle = handle

    def __enter__(self):
        time.sleep(5)
        self.tm_start = time.time()
        self.power_before_gpu = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta_gpu = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000 - self.power_before_gpu
        delta_tm = time.time() - self.tm_start
        
        
        print(f"Time {delta_tm}s Joules: {delta_gpu} Watts: {delta_gpu/delta_tm}")