import argparse
import json
import os
import time

import psutil
import pynvml


def monitor_resources(pid, output_file, stop_file):
    Start_time = time.time_ns()
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except:
        NVML_AVAILABLE = False
        print("pynvml not available. GPU stats will not be shown.")

    process = psutil.Process(pid)
    cpu_usages = []
    ram_usages = []
    gpu_usages = []
    gpu_mem_usages = []
    gpu_power_usages = []

    gpu_handle = None
    if NVML_AVAILABLE:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    if p.pid == pid:
                        gpu_handle = handle
                        break
            except pynvml.NVMLError:
                continue

    while not os.path.exists(stop_file):
        process_cpu_percent = process.cpu_percent(interval=0.1)
        process_memory_info = process.memory_info()
        rss_memory_mb = process_memory_info.rss / (1024 ** 2)

        cpu_usages.append(process_cpu_percent)
        ram_usages.append(rss_memory_mb)

        if NVML_AVAILABLE and gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0

                gpu_usages.append(gpu_util.gpu)
                gpu_mem_usages.append(gpu_mem.used / (1024 ** 2))
                gpu_power_usages.append(power)
            except pynvml.NVMLError:
                pass

        time.sleep(1)
    end_time = time.time_ns()
    duration = (end_time - Start_time) / 1e9
    # Compute averages
    avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
    avg_ram = sum(ram_usages) / len(ram_usages) if ram_usages else 0.0
    avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0.0
    avg_gpu_mem = sum(gpu_mem_usages) / len(gpu_mem_usages) if gpu_mem_usages else 0.0
    avg_gpu_power = sum(gpu_power_usages) / len(gpu_power_usages) if gpu_power_usages else 0.0

    result = {
        "avg_cpu": avg_cpu,
        "avg_cpu_overall": avg_cpu / os.cpu_count(),
        "avg_ram": avg_ram,
        "avg_gpu": avg_gpu,
        "avg_gpu_mem": avg_gpu_mem,
        "avg_gpu_power": avg_gpu_power,
        "monitor_duration_seconds": duration
    }

    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    # print(f"Monitoring results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor")
    parser.add_argument("--output_file", type=str, default="usage.json", help="Output file for usage stats")
    parser.add_argument("--stop_file", type=str, default="stop.txt", help="File to signal stop")
    args = parser.parse_args()

    monitor_resources(args.pid, args.output_file, args.stop_file)
