import subprocess

# List of Python scripts to be executed
scripts = [
    "TCF_ResNet_GenAlg.py",
    "TCF_ResNet_GenAlg_UNFREEZE.py"
]


def run_scripts(script_list):
    for script in script_list:
        print(f"Running {script}...")
        # Run the script and wait for it to complete
        result = subprocess.run(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the script ran successfully
        if result.returncode == 0:
            print(f"{script} completed successfully.")
            print("Output:", result.stdout)
        else:
            print(f"Error in running {script}: {result.stderr}")


# Execute the function with the list of scripts
run_scripts(scripts)
