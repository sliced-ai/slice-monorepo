import subprocess
import sys
import datetime
import json
import os
import signal
import time

LOG_DIR = "/workspace/logs"  # Update this to your desired log folder location

def log_output(logfile, process, log_entries):
    with open(logfile, 'a') as f:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = {"timestamp": timestamp, "output": line.strip()}
                log_entries.append(log_entry)
                f.write(f"{timestamp} - {line}")
                sys.stdout.write(line)

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpu_util = result.stdout.strip()
        return gpu_util
    except Exception as e:
        return f"Failed to get GPU utilization: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python log_wrapper.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1:]
    script_name = os.path.basename(command[0]).split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"{timestamp}_{script_name}.json")
    
    start_time = datetime.datetime.now()
    
    log_data = {
        "command": ' '.join(command),
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "gpu_utilization": get_gpu_utilization(),
        "logs": []
    }
    
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log_output(logfile, process, log_data["logs"])
        process.wait()
        
        if process.returncode == 0:
            status = "Success"
        else:
            status = f"Failed with return code {process.returncode}"
            
    except KeyboardInterrupt:
        if process:
            process.terminate()
        status = "User canceled (KeyboardInterrupt)"
    except Exception as e:
        status = f"Failed with exception: {e}"
    finally:
        end_time = datetime.datetime.now()
        runtime = end_time - start_time
        
        log_data.update({
            "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": status,
            "runtime": str(runtime)
        })
        
        os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
        with open(logfile, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"\nLog saved to {logfile}")

if __name__ == "__main__":
    main()
