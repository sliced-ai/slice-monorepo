import subprocess
import sys
import os
import shutil

def main():
    if len(sys.argv) < 3:
        print("Usage: python log_wrapper.py <command> --save_location <path>")
        sys.exit(1)

    command = sys.argv[1:]
    try:
        save_index = command.index('--save_location')
        save_location = command[save_index + 1]
        command = command[:save_index]
    except (ValueError, IndexError):
        print("Error: --save_location not provided or missing path.")
        sys.exit(1)

    logs_dir = os.path.join(save_location, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Copy the wrapper script to logs directory
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(logs_dir, os.path.basename(script_path)))

    # Copy the command script to the save location
    command_script_path = os.path.abspath(command[0])
    shutil.copy(command_script_path, os.path.join(logs_dir, os.path.basename(command_script_path)))

    # Execute the command and capture output
    log_file_path = os.path.join(logs_dir, 'execution.log')
    with open(log_file_path, 'w') as log_file:
        result = subprocess.run(command + ['--save_location', save_location], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_file.write(result.stdout)
        print(result.stdout)

    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
