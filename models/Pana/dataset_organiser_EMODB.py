import os
import shutil


def organize_emodb_dataset(source_directory, target_base_directory):
    """
    Copies and organizes audio files from the EMODB dataset into folders based on their emotion classes
    in a separate target directory.

    :param source_directory: The directory containing the audio files to organize.
    :param target_base_directory: The base directory where the organized files should be copied to.
    """
    # Mapping of single-letter codes to emotion classes
    emotion_map = {
        "W": "anger",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happiness",
        "T": "sadness",
        "N": "neutral",
    }

    # Ensure source directory ends with a slash
    if not source_directory.endswith('/'):
        source_directory += '/'

    # Ensure target base directory ends with a slash
    if not target_base_directory.endswith('/'):
        target_base_directory += '/'

    # Scan the directory for all .wav files
    for filename in os.listdir(source_directory):
        if filename.endswith('.wav'):
            # Determine the emotion class from the filename
            emotion_code = filename[5]  # 6th character, considering index starts at 0
            emotion_class = emotion_map.get(emotion_code, 'unknown')

            # Directory for the emotion class within the target base directory
            target_directory = os.path.join(target_base_directory, emotion_class)

            # Ensure the target directory exists
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            # Copy the file
            shutil.copy2(os.path.join(source_directory, filename), target_directory)


# Example usage:
organize_emodb_dataset('C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project '
                       'KV6003BNN01/datasets/Vectors/EMODB/wav', 'C:/Users/Pana/Desktop/Northumbria/Final '
                                                                 'Year/Individual Computing Project '
                                                                 'KV6003BNN01/datasets/Mixed/EMODB/')
