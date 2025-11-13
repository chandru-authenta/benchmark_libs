from PIL import Image
import os
import shutil
import io

def copy_images(src_dir, dest_dir, image_extensions=None):
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png'}

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith('.'):
            dest_folder_path = os.path.join(dest_dir,folder)
            if not os.path.exists(dest_folder_path):
                os.makedirs(dest_folder_path)
            print(f"Processing folder: {folder_path}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.startswith('.'):
                    continue
                if not file.lower().endswith(tuple(image_extensions)):
                    continue
                try:
                    shutil.copy2(file_path, dest_folder_path)
                    dest_file_path = os.path.join(dest_folder_path, file)
                    print(f"Copied and converted: {file_path} to {dest_file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    source_directory = "./dataset3"
    destination_directory = "./original_dataset"
    copy_images(source_directory, destination_directory)
    print("Image copying and conversion completed.")