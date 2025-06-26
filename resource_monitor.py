import argparse
import time
import os
import json
import psutil
import subprocess
import re

def main():
    parser = argparse.ArgumentParser(description="Monitor resource usage of a process. On Jetson, uses tegrastats for system/GPU metrics.")
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

    # Data structures for tegrastats
    tegrastats_gpu_util_readings = []
    tegrastats_ram_system_used_readings = []
    # Power readings - add more as needed from parsing
    tegrastats_power_vdd_in_mw_readings = []
    tegrastats_power_cpu_gpu_cv_mw_readings = []
    tegrastats_power_soc_mw_readings = []
    # Temperature readings
    tegrastats_temp_gpu_c_readings = []
    tegrastats_temp_cpu_c_readings = [] # Example for one CPU temp, tegra might have more

    tegrastats_available = True # Assume available, will be set to False if command fails

    print("Monitoring started. Waiting for stop.txt to appear...")
    print("Attempting to use tegrastats for GPU and system metrics.")
    monitoring_duration_seconds = 0

# --- Tegrastats Functions ---
# Example tegrastats output line for parsing reference:
# RAM 1800/7761MB (lfb 2x4MB) CPU [1%@1190,0%@1190,0%@1190,0%@1190,0%@1190,1%@1190] EMC_FREQ 0% GR3D_FREQ 0% AO@32.5C GPU@32C BCPU@32.5C MCPU@32.5C PLL@32.5C Tboard@30C Tdiode@30.75C PMIC@100C thermal@32.2C VDD_IN 3875/3875 VDD_CPU_GPU_CV 874/874 VDD_SOC 1037/1037
# JetPack 5.x example:
# RAM 3480/15691MB (lfb 3x2048kB) SWAP 0/7845MB (cached 0MB) IRAM 0/255kB(lfb 255kB) CPU [0%@1036,0%@1036,0%@1036,0%@1036,0%@1036,0%@1036,0%@1036,0%@1036] EMC_FREQ 0% GR3D_FREQ 0% VIC_FREQ 0% MSENC 0% NVDEC 0% APE 0% GR3D 0W/0W CPU_TOT 0W/0W SOC_TOT 0W/0W CV_TOT 0W/0W VDDRQ 0W/0W SYS5V 0W/0W thermal@30.0C POM_5V_CPU_GPU_CV 0mW/0mW POM_5V_IN 0mW/0mW POM_5V_SOC 0mW/0mW

def parse_tegrastats_output(output_str):
    metrics = {}
    if not output_str:
        return metrics

    # GPU Utilization (GR3D_FREQ X% or GR3D XW/YW)
    # Prefer XW/YW if available (JetPack 5+) as it's actual power/load
    gr3d_power_match = re.search(r"GR3D\s+(\d+)W/(\d+)W", output_str)
    if gr3d_power_match:
        metrics['gpu_power_gr3d_mw'] = int(gr3d_power_match.group(1)) * 1000 # Current power
        # metrics['gpu_load_gr3d_percent'] = (int(gr3d_power_match.group(1)) / int(gr3d_power_match.group(2))) * 100 if int(gr3d_power_match.group(2)) > 0 else 0
    else: # Fallback to GR3D_FREQ % for older JetPacks
        gr3d_freq_match = re.search(r"GR3D_FREQ\s+(\d+)%", output_str)
        if gr3d_freq_match:
            metrics['gpu_util_gr3d_freq_percent'] = int(gr3d_freq_match.group(1))

    # RAM Usage (System Total) - Example: RAM 2365/15829MB
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", output_str)
    if ram_match:
        metrics['ram_system_used_mb'] = int(ram_match.group(1))
        metrics['ram_system_total_mb'] = int(ram_match.group(2))

    # Power VDD_IN (Main input power) - Example: VDD_IN 3340/3340 (current/avg mW)
    vdd_in_match = re.search(r"VDD_IN\s+(\d+)/(\d+)", output_str)
    if vdd_in_match:
        metrics['power_vdd_in_mw'] = int(vdd_in_match.group(1))

    # Power VDD_CPU_GPU_CV - Example: VDD_CPU_GPU_CV 742/742
    vdd_cpu_gpu_cv_match = re.search(r"VDD_CPU_GPU_CV\s+(\d+)/(\d+)", output_str)
    if vdd_cpu_gpu_cv_match:
        metrics['power_cpu_gpu_cv_mw'] = int(vdd_cpu_gpu_cv_match.group(1))

    # Power POM_5V_CPU_GPU_CV (JetPack 5+)
    pom_cpu_gpu_cv_match = re.search(r"POM_5V_CPU_GPU_CV\s+(\d+)mW/(\d+)mW", output_str)
    if pom_cpu_gpu_cv_match:
        metrics['power_cpu_gpu_cv_mw'] = int(pom_cpu_gpu_cv_match.group(1))


    # Power VDD_SOC - Example: VDD_SOC 922/922
    vdd_soc_match = re.search(r"VDD_SOC\s+(\d+)/(\d+)", output_str)
    if vdd_soc_match:
        metrics['power_soc_mw'] = int(vdd_soc_match.group(1))

    # Power POM_5V_SOC (JetPack 5+)
    pom_soc_match = re.search(r"POM_5V_SOC\s+(\d+)mW/(\d+)mW", output_str)
    if pom_soc_match:
        metrics['power_soc_mw'] = int(pom_soc_match.group(1))


    # Temperatures - Example: GPU@28.5C, thermal@29.85C
    # Prefer 'thermal' if available as it's often a summarized critical temperature
    thermal_temp_match = re.search(r"thermal@([\d\.]+)C", output_str)
    if thermal_temp_match:
        metrics['temp_thermal_c'] = float(thermal_temp_match.group(1))

    gpu_temp_match = re.search(r"GPU@([\d\.]+)C", output_str)
    if gpu_temp_match:
        metrics['temp_gpu_c'] = float(gpu_temp_match.group(1))

    cpu_temp_match = re.search(r"(BCPU|MCPU|CPU)@([\d\.]+)C", output_str) # Matches BCPU@, MCPU@ or CPU@
    if cpu_temp_match: # Takes the first CPU temperature it finds (BCPU, MCPU, or generic CPU)
        metrics['temp_cpu_c'] = float(cpu_temp_match.group(2))

    return metrics

def get_tegrastats_snapshot():
    global tegrastats_available # To modify the global flag
    if not tegrastats_available: # If already marked unavailable, don't try again
        return {}

    try:
        # Using --interval 100 and --duration 150 to get a quick single reliable output line.
        # Timeout for communicate() is set to 1 second.
        process = subprocess.Popen(['tegrastats', '--interval', '100', '--duration', '150'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=1)

        if process.returncode != 0:
            # tegrastats might return non-zero if duration is very short, but still print output.
            # So, we check stderr primarily for critical errors.
            if stderr and "tegrastats: command not found" in stderr: # More specific check
                print("Error: 'tegrastats' command not found. GPU/System monitoring will be disabled.")
                tegrastats_available = False
                return {}
            elif stderr:
                 print(f"Tegrastats stderr (return code {process.returncode}): {stderr.strip()}")


        if stdout:
            lines = stdout.strip().splitlines()
            if lines:
                # The last line is usually the full stats line, especially with short duration.
                return parse_tegrastats_output(lines[-1])
        return {} # No output
    except subprocess.TimeoutExpired:
        print("Tegrastats command timed out. GPU/System monitoring will be disabled for this cycle.")
        return {} # Return empty if timed out
    except FileNotFoundError:
        print("Error: 'tegrastats' command not found. Is JetPack installed correctly? GPU/System monitoring disabled.")
        tegrastats_available = False # Mark as unavailable permanently
        return {}
    except Exception as e:
        print(f"Error reading tegrastats: {e}. GPU/System monitoring will be disabled for this cycle.")
        return {} # Return empty on other errors

# --- End Tegrastats Functions ---

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

            # Get tegrastats metrics
            tegra_metrics = get_tegrastats_snapshot()
            if tegra_metrics:
                if 'gpu_util_gr3d_freq_percent' in tegra_metrics:
                    tegrastats_gpu_util_readings.append(tegra_metrics['gpu_util_gr3d_freq_percent'])
                elif 'gpu_power_gr3d_mw' in tegra_metrics: # Alternative GPU metric
                     # If we have power, maybe we prefer to log that instead of creating a new list
                     # For now, let's assume we want the GR3D_FREQ % if available, else nothing for this list
                     pass
                if 'ram_system_used_mb' in tegra_metrics:
                    tegrastats_ram_system_used_readings.append(tegra_metrics['ram_system_used_mb'])
                if 'power_vdd_in_mw' in tegra_metrics:
                    tegrastats_power_vdd_in_mw_readings.append(tegra_metrics['power_vdd_in_mw'])
                if 'power_cpu_gpu_cv_mw' in tegra_metrics:
                    tegrastats_power_cpu_gpu_cv_mw_readings.append(tegra_metrics['power_cpu_gpu_cv_mw'])
                if 'power_soc_mw' in tegra_metrics:
                    tegrastats_power_soc_mw_readings.append(tegra_metrics['power_soc_mw'])
                if 'temp_gpu_c' in tegra_metrics:
                    tegrastats_temp_gpu_c_readings.append(tegra_metrics['temp_gpu_c'])
                if 'temp_cpu_c' in tegra_metrics: # Generic CPU temp from tegra
                    tegrastats_temp_cpu_c_readings.append(tegra_metrics['temp_cpu_c'])
                # Add appends for other parsed tegra metrics if needed

            time.sleep(1.0)
            monitoring_duration_seconds += 1

    except KeyboardInterrupt:
        print("Monitoring interrupted by user.")
    # No finally block needed for pynvml anymore

    results = {}
    # Process-specific metrics from psutil
    if cpu_readings:
        results['process_cpu_avg_percent'] = sum(cpu_readings) / len(cpu_readings)
    else:
        results['process_cpu_avg_percent'] = 0 # Default if no readings

    if ram_readings_mb:
        results['process_ram_avg_mb'] = sum(ram_readings_mb) / len(ram_readings_mb)
    else:
        results['process_ram_avg_mb'] = 0 # Default if no readings

    # Tegrastats metrics
    if tegrastats_gpu_util_readings:
        results['tegrastats_gpu_avg_util_percent'] = sum(tegrastats_gpu_util_readings) / len(tegrastats_gpu_util_readings)
    else:
        results['tegrastats_gpu_avg_util_percent'] = None # Or 0, if preferred for missing data

    if tegrastats_ram_system_used_readings:
        results['tegrastats_ram_system_avg_used_mb'] = sum(tegrastats_ram_system_used_readings) / len(tegrastats_ram_system_used_readings)
    else:
        results['tegrastats_ram_system_avg_used_mb'] = None

    if tegrastats_power_vdd_in_mw_readings:
        results['tegrastats_power_vdd_in_avg_mw'] = sum(tegrastats_power_vdd_in_mw_readings) / len(tegrastats_power_vdd_in_mw_readings)
    else:
        results['tegrastats_power_vdd_in_avg_mw'] = None

    if tegrastats_power_cpu_gpu_cv_mw_readings:
        results['tegrastats_power_cpu_gpu_cv_avg_mw'] = sum(tegrastats_power_cpu_gpu_cv_mw_readings) / len(tegrastats_power_cpu_gpu_cv_mw_readings)
    else:
        results['tegrastats_power_cpu_gpu_cv_avg_mw'] = None

    if tegrastats_power_soc_mw_readings:
        results['tegrastats_power_soc_avg_mw'] = sum(tegrastats_power_soc_mw_readings) / len(tegrastats_power_soc_mw_readings)
    else:
        results['tegrastats_power_soc_avg_mw'] = None

    if tegrastats_temp_gpu_c_readings:
        results['tegrastats_temp_gpu_avg_c'] = sum(tegrastats_temp_gpu_c_readings) / len(tegrastats_temp_gpu_c_readings)
    else:
        results['tegrastats_temp_gpu_avg_c'] = None

    if tegrastats_temp_cpu_c_readings:
        results['tegrastats_temp_cpu_avg_c'] = sum(tegrastats_temp_cpu_c_readings) / len(tegrastats_temp_cpu_c_readings)
    else:
        results['tegrastats_temp_cpu_avg_c'] = None

    results['monitoring_duration_seconds'] = monitoring_duration_seconds
    results['tegrastats_was_available'] = tegrastats_available # Record if tegrastats worked

    output_filename = "resource_usage.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Monitoring results saved to {output_filename}")
    except IOError:
        print(f"Error: Could not write results to {output_filename}")

if __name__ == "__main__":
    main()
