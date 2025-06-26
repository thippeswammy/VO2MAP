import argparse
import time
import os
import json
import psutil
# Attempt to import pynvml and set a flag
try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
                       nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, \
                       NVMLError
    pynvml_available = True
except ImportError:
    pynvml_available = False
    print("pynvml library not found. GPU monitoring will be disabled.")

def get_gpu_metrics(handle):
    metrics = {}
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        metrics['gpu_util_percent'] = util.gpu
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        metrics['gpu_mem_used_mb'] = mem_info.used / (1024**2)
        try:
            power_mw = nvmlDeviceGetPowerUsage(handle)
            metrics['gpu_power_watts'] = power_mw / 1000.0
        except NVMLError: # Some GPUs might not support power reading
            metrics['gpu_power_watts'] = None
    except NVMLError as e:
        print(f"Warning: Could not get some GPU metrics for handle: {e}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Monitor resource usage of a process.")
    parser.add_argument("pid", type=int, help="Process ID to monitor.")
    args = parser.parse_args()

    print(f"Monitoring PID: {args.pid}")

    try:
        process = psutil.Process(args.pid)
        # Initial call to cpu_percent to establish baseline, non-blocking.
        process.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        print(f"Error: Process with PID {args.pid} not found.")
        return
    except psutil.AccessDenied:
        print(f"Error: Access denied to process with PID {args.pid}. Try with sudo?")
        return
    except Exception as e:
        print(f"Error initializing process handle for PID {args.pid}: {e}")
        return


    cpu_readings = []
    ram_readings_mb = []

    gpu_handles = []
    # This will be a list of dicts, where each dict stores lists of readings for a specific GPU
    # e.g., [{'gpu_util_percent': [], 'gpu_mem_used_mb': [], 'gpu_power_watts': []}, ...]
    gpu_metrics_data_list = []


    if pynvml_available:
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            if device_count == 0:
                print("No NVIDIA GPU detected by pynvml.")
                pynvml_available = False
            else:
                print(f"Found {device_count} NVIDIA GPU(s).")
                for i in range(device_count):
                    try:
                        handle = nvmlDeviceGetHandleByIndex(i)
                        gpu_handles.append(handle)
                        gpu_metrics_data_list.append({
                            'gpu_util_percent': [],
                            'gpu_mem_used_mb': [],
                            'gpu_power_watts': []
                        })
                    except NVMLError as e:
                        print(f"Error getting handle for GPU {i}: {e}")
        except NVMLError as e:
            print(f"Error initializing NVML: {e}")
            pynvml_available = False

    print("Monitoring started. Waiting for stop.txt to appear...")
    monitoring_duration_seconds = 0

    try:
        while True:
            if os.path.exists("stop.txt"):
                print("stop.txt detected. Finalizing monitoring.")
                break

            current_cpu_percent = None
            current_ram_mb = None

            try:
                if process.is_running():
                    current_cpu_percent = process.cpu_percent(interval=None)
                    current_ram_mb = process.memory_info().rss / (1024 * 1024)
                else:
                    print(f"Process {args.pid} is no longer running.")
                    break
            except psutil.NoSuchProcess:
                print(f"Process {args.pid} terminated unexpectedly.")
                break
            except Exception as e:
                print(f"Error reading CPU/RAM usage: {e}")
                # Depending on the error, you might want to break or continue
                time.sleep(1.0) # Avoid busy-looping on persistent errors
                continue

            if current_cpu_percent is not None:
                cpu_readings.append(current_cpu_percent)
            if current_ram_mb is not None:
                ram_readings_mb.append(current_ram_mb)

            if pynvml_available and gpu_handles:
                for i, handle in enumerate(gpu_handles):
                    try:
                        metrics = get_gpu_metrics(handle)
                        if 'gpu_util_percent' in metrics:
                           gpu_metrics_data_list[i]['gpu_util_percent'].append(metrics['gpu_util_percent'])
                        if 'gpu_mem_used_mb' in metrics:
                           gpu_metrics_data_list[i]['gpu_mem_used_mb'].append(metrics['gpu_mem_used_mb'])
                        if 'gpu_power_watts' in metrics and metrics['gpu_power_watts'] is not None:
                           gpu_metrics_data_list[i]['gpu_power_watts'].append(metrics['gpu_power_watts'])
                    except NVMLError as e:
                        print(f"Warning: NVML error during GPU metric collection for GPU {i}: {e}")


            time.sleep(1.0)
            monitoring_duration_seconds += 1

    except KeyboardInterrupt:
        print("Monitoring interrupted by user.")
    finally:
        if pynvml_available:
            try:
                # Check if nvmlInit was successfully called by checking if device_count was set (or handles exist)
                # This avoids calling nvmlShutdown if nvmlInit failed.
                if 'device_count' in locals() and nvmlDeviceGetCount() > 0:
                     nvmlShutdown()
            except NameError: # device_count might not be defined if nvmlInit failed early
                pass
            except NVMLError as e:
                print(f"Error during nvmlShutdown: {e}")

    results = {}
    if cpu_readings:
        results['cpu_avg_percent'] = sum(cpu_readings) / len(cpu_readings)
    else:
        results['cpu_avg_percent'] = 0

    if ram_readings_mb:
        results['ram_avg_mb'] = sum(ram_readings_mb) / len(ram_readings_mb)
    else:
        results['ram_avg_mb'] = 0

    # Initialize GPU results with None
    results['gpus_avg_metrics'] = []

    if pynvml_available and gpu_handles:
        for i in range(len(gpu_handles)):
            gpu_avg_data = {
                'gpu_index': i,
                'avg_util_percent': None,
                'avg_mem_mb': None,
                'avg_power_watts': None
            }
            if gpu_metrics_data_list[i]['gpu_util_percent']:
                gpu_avg_data['avg_util_percent'] = sum(gpu_metrics_data_list[i]['gpu_util_percent']) / len(gpu_metrics_data_list[i]['gpu_util_percent'])
            if gpu_metrics_data_list[i]['gpu_mem_used_mb']:
                gpu_avg_data['avg_mem_mb'] = sum(gpu_metrics_data_list[i]['gpu_mem_used_mb']) / len(gpu_metrics_data_list[i]['gpu_mem_used_mb'])
            if gpu_metrics_data_list[i]['gpu_power_watts']: # Check if list is not empty
                gpu_avg_data['avg_power_watts'] = sum(gpu_metrics_data_list[i]['gpu_power_watts']) / len(gpu_metrics_data_list[i]['gpu_power_watts'])
            results['gpus_avg_metrics'].append(gpu_avg_data)

    results['monitoring_duration_seconds'] = monitoring_duration_seconds

    output_filename = "resource_usage.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Monitoring results saved to {output_filename}")
    except IOError:
        print(f"Error: Could not write results to {output_filename}")

if __name__ == "__main__":
    main()
