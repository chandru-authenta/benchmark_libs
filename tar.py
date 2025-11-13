import tarfile
import os
import random
import json
import io

dataset_folder = "./original_dataset"
output_folder = "./tar_shards"
os.makedirs(output_folder, exist_ok=True)

subfolders = [
	os.path.join(dataset_folder, d)
	for d in os.listdir(dataset_folder)
	if os.path.isdir(os.path.join(dataset_folder, d))
]

all_images=[]
all_labels=[]
count=0
for subfolder in subfolders:
    for fname in os.listdir(subfolder):
        if fname.startswith('.') or not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(subfolder, fname)
        all_images.append(path)
        count += 1
        all_labels.append(f'{subfolder}/{fname}/{count}')
# random.shuffle(all_images)
combined = list(zip(all_images,all_labels))
random.shuffle(combined)
all_images, all_labels = zip(*combined)

print(f"---Total images found: {len(all_images)}")

shard_size = 2000
num_shards = (len(all_images) + shard_size - 1) // shard_size


for shard_idx in range(num_shards):
    start_idx = shard_idx * shard_size
    end_idx = min((shard_idx + 1) * shard_size, len(all_images))
    shard_images = all_images[start_idx:end_idx]
    shard_labels = all_labels[start_idx:end_idx]

    shard_tar_path = os.path.join(
        output_folder,
        f"shard_{str(shard_idx + 1).zfill(2)}.tar"
    )
    with tarfile.open(shard_tar_path, "w") as tar:
        for image_path, label in zip(shard_images, shard_labels):
            base = os.path.splitext(os.path.basename(image_path))[0]
            arc_name = f"{base}.png"
            tar.add(image_path, arcname=arc_name)

            # metadata = {"label": label}
            # data = json.dumps(metadata).encode("utf-8")
            # info = tarfile.TarInfo(name=f"{base}.json")
            # info.size = len(data)
            # tar.addfile(info, io.BytesIO(data))
    print(f"Created shard: {shard_tar_path} with {len(shard_images)} images")

