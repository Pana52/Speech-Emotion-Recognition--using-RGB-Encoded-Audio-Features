import subprocess

# List of Python scripts to be executed
scripts = [
    "TCF_ResNet_GenAlg_RAVDESS_01.py",
    "TCF_ResNet_GenAlg_RAVDESS_02.py",
    "TCF_ResNet_GenAlg_RAVDESS_03.py",
    "TCF_ResNet_GenAlg_RAVDESS_04.py",
    "TCF_ResNet_GenAlg_RAVDESS_05.py",
    "TCF_ResNet_GenAlg_RAVDESS_06.py",
    # Add more scripts as needed
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

        # Optionally, add a pause or delay if needed
        # import time
        # time.sleep(seconds)  # seconds between runs, if needed


# Execute the function with the list of scripts
run_scripts(scripts)
