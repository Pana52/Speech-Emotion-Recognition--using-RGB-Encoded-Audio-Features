# Re-import necessary modules after reset
import os
import shutil


def organize_ravdess_dataset(source_directory, target_base_directory):
    """
    Recursively copies and organizes audio files from the RAVDESS dataset, structured by actors,
    into folders based on their emotion classes in a separate target directory.

    :param source_directory: The root directory containing the RAVDESS audio files, structured by actor directories.
    :param target_base_directory: The base directory where the organized files should be copied to.
    """
    # Mapping of numerical emotion codes to emotion classes
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised',
    }

    # Walk through source directory
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith('.wav'):
                # Extract emotion code from the filename
                parts = filename.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion_class = emotion_map.get(emotion_code, 'unknown')

                    # Directory for the emotion class within the target base directory
                    target_directory = os.path.join(target_base_directory, emotion_class)

                    # Ensure the target directory exists
                    if not os.path.exists(target_directory):
                        os.makedirs(target_directory)

                    # Copy the file
                    shutil.copy2(os.path.join(root, filename), target_directory)


# Example usage:
# organize_ravdess_dataset_v2('/path/to/your/ravdess/dataset', '/path/to/target/directory')


organize_ravdess_dataset('C:/Users/Pana/Desktop/Northumbria/Final Year/Individual Computing Project '
                         'KV6003BNN01/datasets/Vectors/RAVDESS/', 'C:/Users/Pana/Desktop/Northumbria/Final '
                                                                   'Year/Individual Computing Project '
                                                                   'KV6003BNN01/datasets/Mixed/RAVDESS/')
