import subprocess

# List of Python scripts to be executed
scripts = [
    "TCF_CNN_GenAlg.py",
    "TFC_DenseNet_GenAlg_FREEZE.py",
    "TFC_DenseNet_GenAlg_UNFFREEZE.py",
    "TCF_EfficientNet_GenAlg_FREEZE.py",
    "TCF_EfficientNet_GenAlg_UNFREEZE.py",
    "TCF_ResNet_GenAlg_FREEZE.py",
    "TCF_ResNet_GenAlg_UNFREEZE.py",
    "TCF_VGG_GenAlg_FREEZE.py",
    "TCF_VGG_GenAlg_UNFREEZE.py",
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
