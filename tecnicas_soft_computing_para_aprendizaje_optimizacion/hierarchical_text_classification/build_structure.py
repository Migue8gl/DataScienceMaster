import os
import shutil


def build_structure():
    """
    Creates the necessary project folder structure ('data', 'img', 'doc', 'src') if they do not exist.
    Moves files from the current directory into the appropriate folders based on their extensions:
    - .csv files to 'data'
    - .png and .jpg files to 'img'
    - .pdf files to 'doc'
    """
    folders = ["data", "img", "doc", "models"]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if os.path.isdir(filename):
            continue

        file_path = os.path.join(current_dir, filename)

        if filename.endswith(".csv") or filename.endswith(".json"):
            shutil.move(file_path, os.path.join(current_dir, "data", filename))
            print(f"Moved {filename} to data/")

        elif filename.endswith(".png") or filename.endswith(".jpg"):
            shutil.move(file_path, os.path.join(current_dir, "img", filename))
            print(f"Moved {filename} to img/")

        elif filename.endswith(".pdf"):
            shutil.move(file_path, os.path.join(current_dir, "doc", filename))
            print(f"Moved {filename} to doc/")


if __name__ == "__main__":
    build_structure()
