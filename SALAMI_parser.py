import os

folder_path = "./datasets/SALAMI/audio"

"""
# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an mp3 file
    if filename.endswith(".mp3"):
        # Create a new filename with the "SALAMI_" prefix
        new_filename = "SALAMI_" + filename

        # Get the full path of the original and new file
        original_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(original_file_path, new_file_path)

print("All mp3 files have been renamed.")
"""

for filename in os.listdir(folder_path):
    if filename.startswith("SALAMI_SALAMI_"):
        new_filename = filename.replace("SALAMI_SALAMI_", "SALAMI_")
        os.rename(
            os.path.join(folder_path, filename), os.path.join(folder_path, new_filename)
        )
