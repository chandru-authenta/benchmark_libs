
from streaming import MDSWriter
from PIL import Image
import os
import io
import pathlib
from multiprocessing import Pool,cpu_count
from tqdm import tqdm

input_folder = './original_dataset'
output_dir = './output_shards2'
# BATCH_SIZE= 32
# NUM_WORKERS = 4
if not os.path.exists(input_folder):
    exit(f"Input folder {input_folder} does not exist.")

if not os.path.exists(output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

columns = {
    'image': 'jpeg',
    'label': 'int'
}

# def load_and_convert_image(file_info):
#     path, class_idx = file_info
#     try:
#         img=Image.open(path).convert('RGB')
#         img_io=io.BytesIO()
#         img.save(img_io, format="JPEG", quality=95, optimize=True)
#         img_io.seek(0)
#         return {'image': img_io.getvalue(), 'label': class_idx}
#     except Exception as e:
#         print(f"Error processing {path}: {e}")
#         return None

# file_list = []
# class_map ={}
# for class_idx, class_name in enumerate(sorted(os.listdir(input_folder))):
#     if class_name.startswith('.'):
#         continue
#     class_folder = os.path.join(input_folder, class_name)
#     if not os.path.isdir(class_folder):
#         continue
#     class_map[class_idx] = class_name
#     for fname in os.listdir(class_folder):
#         if fname.startswith('.') or not fname.lower().endswith(('.png')):
#             continue
#         file_list.append((os.path.join(class_folder, fname), class_idx))

with MDSWriter(out=output_dir, columns=columns, compression='zstd',size_limit='200mb') as writer:
    # with Pool(NUM_WORKERS) as pool:
    #     for batch_idx in tqdm(range(0, len(file_list), BATCH_SIZE), desc="Processing batches"):
    #         batch = file_list[batch_idx:batch_idx + BATCH_SIZE]
    #         results = pool.map(load_and_convert_image, batch)
    #         for sample in results:
    #             if sample is not None:
    #                 writer.write(sample)
    for class_idx, class_name in enumerate(os.listdir(input_folder)):
        if class_name.startswith('.'):
            continue  
        class_folder = os.path.join(input_folder, class_name)
        print(f"Processing class '{class_name}' with label {class_idx}")
        for fname in os.listdir(class_folder):
            if fname.startswith('.'):
                continue  
            if not fname.lower().endswith(('.png')):
                continue  
            print(f"  Adding file: {class_folder}-> {fname}, label: {class_idx}")
            path = os.path.join(class_folder, fname)
            img = Image.open(path).convert('RGB')
            sample = {'image': img, 'label': class_idx}
            writer.write(sample)

print("✓ Dataset sharding completed successfully!")
                    
