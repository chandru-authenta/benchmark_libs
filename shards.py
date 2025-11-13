import tarfile
import os
import random
import json
# from StorageUtils import StorageUtils
import io
dataset_folder = "./dataset"
output_folder = "tar_shards"
# storage = StorageUtils(output_folder)
os.makedirs(output_folder, exist_ok=True)

subfolders = [
	os.path.join(dataset_folder, d)
	for d in os.listdir(dataset_folder)
	if os.path.isdir(os.path.join(dataset_folder, d))
]


fake_data, real_data = sorted(subfolders)


real_images = [os.path.join(real_data, f) for f in os.listdir(real_data)]
fake_images = [os.path.join(fake_data, f) for f in os.listdir(fake_data)]

real_lables = [f"real{str(i+1).zfill(5)}" for i in range(len(real_images))]
fake_lables = [f"fake{str(i+1).zfill(5)}" for i in range(len(fake_images))]

all_images = real_images + fake_images
all_labels = real_lables + fake_lables

combined = list(zip(all_images, all_labels))
random.shuffle(combined)
all_images, all_labels = zip(*combined)

shard_size = 2000 # Number of images per shard
num_shards = (len(all_images) + shard_size - 1) // shard_size

for shard_idx in range(num_shards):
    start_idx = shard_idx * shard_size
    end_idx = min((shard_idx + 1) * shard_size, len(all_images))
    shard_images = all_images[start_idx:end_idx]
    shard_labels = all_labels[start_idx:end_idx]
    
    shard_tar_path = os.path.join(
        output_folder,
        f"shared_{str(shard_idx + 1).zfill(2)}.tar"
    )
    with tarfile.open(shard_tar_path, 'w') as tar:
        for img_path, label in zip(shard_images, shard_labels):
            # Add image
            base = os.path.splitext(os.path.basename(img_path))[0]
            arcname_img = f"{base}.png"
            tar.add(img_path, arcname=arcname_img)
            
            # Add metadata (json)
            metadata = {"label": "real" if label.startswith("real") else "fake"}
            data = json.dumps(metadata).encode("utf-8")
            info = tarfile.TarInfo(name=f"{base}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            
    # file_size = storage.format_size(storage.get_file_size(shard_tar_path))
    print(f"Created {shard_tar_path} of size: && {shard_idx+1}/{num_shards} with {len(shard_images)} images.")
    
print("Sharding completed!")
# total, file_sizes = storage.total_folder_size()
# print("Shards Folder Size: ", storage.format_size(total))
# print("dataset folder size: ", StorageUtils(dataset_folder))