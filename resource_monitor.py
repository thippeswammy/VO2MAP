import argparse
import time
import os
import json
import psutil
import threading

try:
    from jtop import JTop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False

def extract_val(obj):
    if isinstance(obj, dict):
        return obj.get("val")
    elif isinstance(obj, (int, float)):
        return obj
    return None

def monitor(pid, output_file="resource_usage.json", stop_file="stop.txt"):
    process = psutil.Process(pid)
    cpu_readings, ram_readings = [], []
    gpu_util_readings, gpu_mem_readings, gpu_temp_readings = [], [], []
    power_readings = []

    start_time = time.time()

    def collect_with_jtop():
        with JTop() as jetson:
            while not os.path.exists(stop_file):
                stats = jetson.stats
                cpu_readings.append(process.cpu_percent(interval=0.1))
                ram_readings.append(process.memory_info().rss / 1024 / 1024)

                gpu = extract_val(stats.get("GPU"))
                if gpu is not None:
                    gpu_util_readings.append(gpu)

                mem_gpu = extract_val(stats.get("GPU Memory"))
                if mem_gpu is not None:
                    gpu_mem_readings.append(mem_gpu)

                temp = extract_val(stats.get("Temp GPU"))
                if temp is not None:
                    gpu_temp_readings.append(temp)

                power = extract_val(stats.get("Power GPU"))
                if power is not None:
                    power_readings.append(power)

                time.sleep(1)

    def collect_basic():
        while not os.path.exists(stop_file):
            try:
                cpu_readings.append(process.cpu_percent(interval=0.1))
                ram_readings.append(process.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break
            time.sleep(1)

    if HAS_JTOP:
        collect_with_jtop()
    else:
        collect_basic()

    duration = time.time() - start_time

    stats = {
        "avg_cpu": sum(cpu_readings)/len(cpu_readings) if cpu_readings else 0,
        "avg_cpu_overall": sum(cpu_readings)/len(cpu_readings)/psutil.cpu_count() if cpu_readings else 0,
        "avg_ram": sum(ram_readings)/len(ram_readings) if ram_readings else 0,
        "avg_gpu": sum(gpu_util_readings)/len(gpu_util_readings) if gpu_util_readings else 0,
        "avg_gpu_mem": sum(gpu_mem_readings)/len(gpu_mem_readings) if gpu_mem_readings else 0,
        "avg_gpu_temp": sum(gpu_temp_readings)/len(gpu_temp_readings) if gpu_temp_readings else 0,
        "avg_gpu_power": sum(power_readings)/len(power_readings) if power_readings else 0,
        "monitor_duration_seconds": duration
    }

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)

    with open("resource_monitor.done", 'w') as f:
        f.write("done\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int, help="PID of the VSLAM process to monitor")
    args = parser.parse_args()
    monitor(args.pid)
