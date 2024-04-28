import subprocess


def execute_python_files_sequentially(file_list):

    for file in file_list:
        print(f"Executing {file}...")
        try:
            result = subprocess.run(['python', file], check=True, text=True, capture_output=True)
            print(f"Execution of {file} completed successfully.")
            print(f"Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {file}: {e}")
            print(f"Error output:\n{e.stderr}")
            break  # or continue, depending on whether to stop or skip to the next file on error


# Example file list
files_to_execute = ["TCF_EfficientNet_GenAlg_FREEZE_01.py",
                    "TCF_EfficientNet_GenAlg_FREEZE_02.py",
                    "TCF_EfficientNet_GenAlg_FREEZE_03.py",
                    "TCF_EfficientNet_GenAlg_FREEZE_04.py",
                    "TCF_EfficientNet_GenAlg_FREEZE_05.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_01.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_02.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_03.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_04.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_05.py",
                    "TCF_EfficientNet_GenAlg_FREEZE4C_06.py",
                    ]

# Uncomment the following line to execute the function:
execute_python_files_sequentially(files_to_execute)
