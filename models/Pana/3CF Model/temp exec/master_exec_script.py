import subprocess

# List of Python scripts to be executed
scripts = [
    "TCF_CNN_GenAlg_CH_ME_MF.py",
    "TCF_CNN_GenAlg_CH_MF_ME.py",
    "TCF_CNN_GenAlg_ME_CH_MF.py",
    "TCF_CNN_GenAlg_ME_MF_CH.py",
    "TCF_CNN_GenAlg_MF_CH_ME.py",
    "TCF_CNN_GenAlg_MF_ME_CH.py",
    "TCF_DenseNet_GenAlg_CH_ME_MF.py",
    "TCF_DenseNet_GenAlg_CH_MF_ME.py",
    "TCF_DenseNet_GenAlg_ME_CH_MF.py",
    "TCF_DenseNet_GenAlg_ME_MF_CH.py",
    "TCF_DenseNet_GenAlg_MF_CH_ME.py",
    "TCF_DenseNet_GenAlg_MF_ME_CH.py",
    "TCF_ResNet_GenAlg_CH_ME_MF.py",
    "TCF_ResNet_GenAlg_CH_MF_ME.py",
    "TCF_ResNet_GenAlg_ME_CH_MF.py",
    "TCF_ResNet_GenAlg_ME_MF_CH.py",
    "TCF_ResNet_GenAlg_MF_CH_ME.py",
    "TCF_ResNet_GenAlg_MF_ME_CH.py",
    "TCF_VGG_GenAlg_CH_ME_MF.py",
    "TCF_VGG_GenAlg_CH_MF_ME.py",
    "TCF_VGG_GenAlg_ME_CH_MF.py",
    "TCF_VGG_GenAlg_ME_MF_CH.py",
    "TCF_VGG_GenAlg_MF_CH_ME.py",
    "TCF_VGG_GenAlg_MF_ME_CH.py"
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
