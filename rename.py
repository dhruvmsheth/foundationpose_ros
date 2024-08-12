import os
import re

def rename_files_in_folder(folder_path):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            # Extract only the digits from the filename
            new_filename = ''.join(re.findall(r'\d+', filename)) + '.png'
            # Create the full path for the old and new filenames
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')

# Define the folders
folders = [
    'demo_data/outputs/depth',
    'demo_data/outputs/rgb',
    'demo_data/outputs/masks'
]

# Rename files in each folder
for folder in folders:
    rename_files_in_folder(folder)
