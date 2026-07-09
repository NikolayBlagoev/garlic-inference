import pynvml
import time

RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_PATH    = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"

def _read_rapl(path):
    with open(path) as f:
        return int(f.read().strip())

class PowerProfiler:
    def __init__(self, handle, use_cpu=False):
        self.power_before_gpu = 0
        self.power_before_cpu = 0
        self.rapl_max = 0
        self.tm_start = 0
        self.handle = handle
        self.use_cpu = use_cpu

    def __enter__(self):
        time.sleep(5)
        self.tm_start = time.time()
        self.power_before_gpu = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000
        if self.use_cpu:
            self.rapl_max = _read_rapl(RAPL_MAX_PATH)
            self.power_before_cpu = _read_rapl(RAPL_ENERGY_PATH)

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta_gpu = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000 - self.power_before_gpu
        delta_tm = time.time() - self.tm_start

        cpu_j = 0.0
        if self.use_cpu:
            rapl_after = _read_rapl(RAPL_ENERGY_PATH)
            # handle counter wraparound
            if rapl_after >= self.power_before_cpu:
                delta_uj = rapl_after - self.power_before_cpu
            else:
                delta_uj = self.rapl_max - self.power_before_cpu + rapl_after
            cpu_j = delta_uj / 1e6

        total_j = delta_gpu + cpu_j
        print(f"Time {delta_tm}s GPU Joules: {delta_gpu} CPU Joules: {cpu_j} Total Joules: {total_j} Watts: {total_j/delta_tm}")
