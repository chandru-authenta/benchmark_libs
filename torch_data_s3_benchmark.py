import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision import transforms
from PIL import Image
import s3fs
import io
import os
import numpy as np
import psutil
import time
import csv

class S3ImageStreamDataset(IterableDataset):
    def __init__(self, s3_prefix, transform=None):

        super().__init__()
        self.s3_prefix = s3_prefix.rstrip("/")
        self.fs = s3fs.S3FileSystem(anon=False)  # ensure credentials / role available
        # find all files under prefix (recursive)
        try:
            all_files = self.fs.find(self.s3_prefix)
        except Exception as e:
            # some s3fs versions expect bucket/key without scheme for find; try strip s3://
            if self.s3_prefix.startswith("s3://"):
                without_scheme = self.s3_prefix[5:]
                all_files = self.fs.find(without_scheme)
            else:
                raise
        # filter by extensions
        exts = (".jpg", ".jpeg", ".png", ".webp")
        self.files = sorted([f for f in all_files if f.lower().endswith(exts)])
        self.transform = transform
        print(f"✅ Found {len(self.files)} S3 images under: {self.s3_prefix}")

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
           
            chunks = np.array_split(np.array(self.files), num_workers)
            files = list(chunks[worker_id]) if len(chunks) > worker_id else []

        for key in files:
            try:
                open_key = key
                # If key doesn't start with "s3://" and prefix did, s3fs.open still accepts "bucket/key".
                with self.fs.open(open_key, "rb") as f:
                    img_bytes = f.read()
                if not img_bytes:
                    print(f"⚠️ Skipping empty file: {key}")
                    continue
                # decode with PIL
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                yield img
            except Exception as e:
                # keep running on error, but log filename and error
                print(f"⚠️ Error loading {key}: {repr(e)}")
                continue

    def state_dict(self):
        # no custom state to checkpoint here (worker sharding is deterministic)
        return {}

    def load_state_dict(self, state_dict):
        pass



def run_benchmark(dataset, run_id, num_workers=4, batch_size=32):
    print(f"\n🔄 Running benchmark iteration {run_id + 1}...")
    process = psutil.Process(os.getpid())

    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )

    print("🔎 Measuring dataset loading performance...")
    all_data_load_times = []
    all_cpu, all_ram = [], []
    total_images = 0

    def get_metrics():
        cpu = psutil.cpu_percent(interval=None)
        ram = process.memory_info().rss / (1024**3)
        try:
            disk = psutil.disk_usage("/").used / (1024**3)
        except Exception:
            disk = float("nan")
        return cpu, ram, disk

    start_time = time.time()
    batch_idx = 0
    
    for batch in loader:
        batch_start = time.time()
        elapsed = time.time() - batch_start
        all_data_load_times.append(elapsed)

        batch_idx += 1
        # batch may be a list of images or a tensor; handle both
        try:
            batch_len = len(batch)
        except Exception:
            batch_len = 1
        total_images += batch_len
        print(f"🖼️ processed batch {batch_idx} with {batch_len} images (load took {elapsed:.4f}s)")

        cpu, ram, disk = get_metrics()
        all_cpu.append(cpu)
        all_ram.append(ram)
        # not storing disk per-batch to keep arrays small; optional if you want
    total_time = time.time() - start_time

    # handle empty runs gracefully
    if len(all_data_load_times) == 0:
        avg_load = float("nan")
    else:
        avg_load = float(np.mean(all_data_load_times))

    return {
        "total_time": total_time,
        "total_images": total_images,
        "images_per_sec": (total_images / total_time) if total_time > 0 and total_images > 0 else 0.0,
        "avg_data_load_time": avg_load,
        "avg_cpu": float(np.mean(all_cpu)) if all_cpu else float("nan"),
        "avg_ram": float(np.mean(all_ram)) if all_ram else float("nan"),
    }


# --------------------------
# Main
# --------------------------
def main():
    s3_uri = "s3://authenta-streaming-data/original_dataset/dragon_train_000"
    print(f"Loading dataset from: {s3_uri}")

    num_runs = 1
    print(f"\n🚀 Starting {num_runs} benchmark runs...")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = S3ImageStreamDataset(s3_uri, transform=transform)

    all_metrics = []
    for run_id in range(num_runs):
        metrics = run_benchmark(dataset, run_id, num_workers=4, batch_size=32)
        all_metrics.append(metrics)

        # Print individual run results
        print(f"\n📊 Run {run_id + 1} Results:")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Total images: {metrics['total_images']}")
        print(f"  Images per second: {metrics['images_per_sec']:.2f}")
        print(f"  Avg data load time: {metrics['avg_data_load_time']:.4f}s")

 # Calculate statistics across all runs
    metrics_keys = [
        "total_time",
        "total_images",
        "images_per_sec",
        "avg_data_load_time",
        "avg_cpu",
        "avg_ram",
    ]

    stats = {}
    for key in metrics_keys:
        values = [run[key] for run in all_metrics]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    # Save detailed metrics to CSV
    metrics_file = "metrics/torchdata_metrics_s3_images.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header for individual runs
        writer.writerow(["run_id"] + metrics_keys)

        # Write individual run data
        for i, metrics in enumerate(all_metrics):
            writer.writerow([i + 1] + [metrics[key] for key in metrics_keys])

        # Write empty line and statistics
        writer.writerow([])
        writer.writerow(["statistic"] + metrics_keys)
        writer.writerow(["mean"] + [stats[key]["mean"] for key in metrics_keys])
        writer.writerow(["std"] + [stats[key]["std"] for key in metrics_keys])

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print(f"📈 BENCHMARK SUMMARY STATISTICS ({num_runs} runs)")
    print("=" * 60)

    print("\n🕐 Total Time:")
    mean_time = stats["total_time"]["mean"]
    std_time = stats["total_time"]["std"]
    print(f"  Mean: {mean_time:.2f}s ± {std_time:.2f}s")

    print("\n🚀 Images per Second:")
    mean_ips = stats["images_per_sec"]["mean"]
    std_ips = stats["images_per_sec"]["std"]
    print(f"  Mean: {mean_ips:.2f} ± {std_ips:.2f}")

    print("\n⚡ Average Data Load Time:")
    mean_load = stats["avg_data_load_time"]["mean"]
    std_load = stats["avg_data_load_time"]["std"]
    print(f"  Mean: {mean_load:.4f}s ± {std_load:.4f}s")

    print("\n💾 CPU Usage:")
    mean_cpu = stats["avg_cpu"]["mean"]
    std_cpu = stats["avg_cpu"]["std"]
    print(f"  Mean: {mean_cpu:.1f}% ± {std_cpu:.1f}%")

    print("\n🧠 RAM Usage:")
    mean_ram = stats["avg_ram"]["mean"]
    std_ram = stats["avg_ram"]["std"]
    print(f"  Mean: {mean_ram:.2f}GB ± {std_ram:.2f}GB")

    print(f"\n💾 Results saved to: {metrics_file}")
    

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()



