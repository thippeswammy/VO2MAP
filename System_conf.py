import platform
import psutil
import socket
import subprocess

def get_system_info():
    print("=== SYSTEM INFORMATION ===")
    print(f"Hostname       : {socket.gethostname()}")
    print(f"OS             : {platform.system()} {platform.release()}")
    print(f"OS Version     : {platform.version()}")
    print(f"Architecture   : {platform.machine()}")
    print(f"Processor      : {platform.processor()}")

    cpu_freq = psutil.cpu_freq()
    print(f"CPU Cores      : {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"CPU Frequency  : {cpu_freq.current:.2f} MHz (min: {cpu_freq.min:.2f}, max: {cpu_freq.max:.2f})")

    ram = psutil.virtual_memory()
    print(f"RAM (Total)    : {round(ram.total / (1024**3), 2)} GB")
    print(f"RAM (Available): {round(ram.available / (1024**3), 2)} GB")

def get_disk_info():
    print("\n=== DISK INFORMATION ===")
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        print(f"{partition.device} ({partition.mountpoint}) [{partition.fstype}]")
        print(f"  Total: {round(usage.total / (1024**3), 2)} GB | Used: {round(usage.used / (1024**3), 2)} GB | Free: {round(usage.free / (1024**3), 2)} GB | Usage: {usage.percent}%")

def get_network_info():
    print("\n=== NETWORK INFORMATION ===")
    for interface_name, interface_addresses in psutil.net_if_addrs().items():
        for addr in interface_addresses:
            if str(addr.family) == 'AddressFamily.AF_INET':
                print(f"{interface_name} → IP: {addr.address}, Netmask: {addr.netmask}, Broadcast: {addr.broadcast}")

def get_jetson_gpu_info():
    print("\n=== JETSON GPU/POWER INFO (tegrastats) ===")
    try:
        output = subprocess.check_output(["sudo", "tegrastats", "--interval", "1000"], timeout=2).decode()
        print(output.strip())
    except subprocess.TimeoutExpired:
        print("Captured 1-second snapshot using tegrastats.")
    except Exception as e:
        print("tegrastats failed:", e)

if __name__ == "__main__":
    get_system_info()
    get_disk_info()
    get_network_info()
    get_jetson_gpu_info()
